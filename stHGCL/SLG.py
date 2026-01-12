import dgl
import numpy as np
import scanpy as sc
import torch
from torch import nn, optim
import torch.nn.functional as F
import gc
import dgl.function as fn
from .utils import setup_seed,mclust_R,calculate_metric,kmeans,KMeans
from .Graph_utils import make_graph,graph_construction,generate_adj_list,make_big_graph
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import anndata as ad
from .Mnn_utils import create_dictionary_mnn
from .model_multi import LightGCN_multi
from .model import lightST,target_distribution



def SLGnet_multi(adata, batch_name,
                  Batch_list,
                  n_clusters=7,
                  seed = 666,
                  n_layers=2,
                  feats_dim=64,
                  drop_out=0.1,
                  gamma=1,
                  decoder='Dot',
                  lr=0.1,
                  epoch_1=800,
                  epoch_2=800,
                  log_interval=10,
                  sample_rate=0.1,
                  learnable_w=True,
                  highly_variable=None,
                  recon_ratio=1.,
                  cl_ratio=1.,
                  k=50,
                  neigh=7,
                  alpha=1.,
                  margin=1.0,
                  verbose=False,
                  gradient_clipping=5.,
                  MNN=False,
                  beta=1.
                  ):
    setup_seed(seed)
    section_ids = np.array(adata.obs['batch'].unique())
    n = neigh
    n_batch = len(batch_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj_list = generate_adj_list(Batch_list, n_batch, n, device)
    print('Graph construction is done...')
    single_adata = []
    for i in range(n_batch):
        single_adata.append(adata[adata.obs['batch'].values == batch_name[i]])

    assert decoder in ['Dot', 'GMF', 'ZINB'], "Please choose decoder in ['Dot','GMF','ZINB']"
    assert sample_rate <= 1, "Please set 0<sample_rate<=1"

    ####################   Prepare data for training   ####################

    raw_exp = True if decoder == 'ZINB' else False
    # 用于计算指标的adata
    # embedding_ad = ad.AnnData(adata.X)
    # embedding_ad.obs['celltype'] = cell_type
    # embedding_ad.obs['batch'] = adata.obs['batch'].values
    # embedding_ad.obs_names = [f"Cell_{i:d}" for i in range(embedding_ad.n_obs)]
    # embedding_ad.var_names = adata.var_names

    single_adata, exp_value, exp_enc_graph, exp_graph, exp_val_graph, exp_val_value, unexp_edges = make_big_graph(
        single_adata, n_batch, raw_exp, highly_variable)

    n_pos_edges = []
    n_neg_edges = []
    n_cells = []

    for i in range(n_batch):
        n_pos_edges.append(int(sample_rate * len(exp_value[i])))
        n_neg_edges.append(int(sample_rate * len(unexp_edges[i][0])))

        exp_value[i] = torch.tensor(exp_value[i], device=device)
        exp_val_value[i] = torch.tensor(exp_val_value[i], device=device)
        exp_graph[i] = exp_graph[i].to(device)
        exp_val_graph[i] = exp_val_graph[i].to(device)

        n_cells.append(single_adata[i].X.shape[0])

    n_genes = adata.n_vars
    exp_enc_graph = exp_enc_graph.to(device)

    #######################   Prepare models   #######################

    model = LightGCN_multi(n_layers=n_layers,
                     n_cells=n_cells,
                     n_genes=n_genes,
                     drop_out=drop_out,
                     adj_list=adj_list,
                     decoder=decoder,
                     feats_dim=feats_dim,
                     learnable_weight=learnable_w).to(device)  # 初始细胞和基因的维度
    # print(model)
    model = model.to(device)

    if decoder in ['Dot', 'GMF']:
        criterion = nn.MSELoss()
    else:
        criterion = ZINBLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)  ### weight_decay可以调整

    #######################   Record values   #######################
    all_loss = []
    all_loss_val = []

    # best_ari_l = 0
    # best_nmi_l = 0
    # best_sil_f = 0
    # # no_better_valid = 0
    # all_acc_l = []
    # all_ari_l = []
    # all_nmi_l = []
    # all_sil_f = []
    #
    # best_iter_l = -1

    count_cl_loss = 0

    stop_flag = False
    #######################   Start training model   #######################
    print(f"Start training on {device}...")
    all_unexp_index = []
    # unexp_sample = []
    unexp_dec_graph = []
    exp_dec_graph = []
    exp_dec_value = []
    all_index = []

    for i in range(n_batch):
        all_unexp_index.append(np.arange(len(unexp_edges[i][0])))
        unexp_sample_index = np.random.choice(all_unexp_index[i], n_neg_edges[i])
        unexp_sample = (unexp_edges[i][0][unexp_sample_index], unexp_edges[i][1][unexp_sample_index])
        unexp_dec_graph.append(dgl.heterograph({('cell' + str(i + 1), 'exp' + str(i + 1), 'gene'): unexp_sample},
                                               num_nodes_dict={'cell' + str(i + 1): n_cells[i], 'gene': n_genes}).to(
            device))

        if sample_rate == 1:
            exp_dec_graph.append(exp_graph[i])
            exp_dec_value.append(exp_value[i])
            if decoder == 'ZINB':
                exp_dec_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = \
                exp_graph[i].nodes['cell' + str(i + 1)].data['sz_factor']
                exp_dec_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']
        else:
            all_index.append(np.arange(len(exp_value[i])))

    adata = ad.concat(single_adata, merge="same")

    ridge = [None] * n_batch
    print('Train with SLGnet...')
    for iter_idx in tqdm(range(0, epoch_1)):

        if sample_rate != 1:
            exp_dec_value = []
            exp_dec_graph = []
            for i in range(n_batch):
                exp_sample_index = np.random.choice(all_index[i], n_pos_edges[i])
                exp_dec_value.append(exp_value[i][exp_sample_index])
                exp_sample = (exp_graph[i].edges()[0][exp_sample_index], exp_graph[i].edges()[1][exp_sample_index])

                exp_dec_graph.append(dgl.heterograph({('cell' + str(i + 1), 'exp' + str(i + 1), 'gene'): exp_sample},
                                                     num_nodes_dict={'cell' + str(i + 1): n_cells[i], 'gene': n_genes})
                                     )

                if decoder == 'ZINB':
                    exp_dec_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = \
                    exp_graph[i].nodes['cell' + str(i + 1)].data['sz_factor']
                    exp_dec_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']

        if decoder == 'ZINB':
            for i in range(n_batch):
                unexp_dec_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = \
                exp_graph[i].nodes['cell' + str(i + 1)].data['sz_factor']
                unexp_dec_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']
                exp_val_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = \
                exp_graph[i].nodes['cell' + str(i + 1)].data['sz_factor']
                exp_val_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']

        pred_exp, pred_unexp = model(exp_enc_graph, exp_dec_graph, adj_list, unexp_dec_graph)

        if decoder in ['Dot', 'GMF']:
            # loss_exp = criterion(pred_exp, exp_dec_value)
            # loss_unexp = criterion(pred_unexp, torch.zeros_like(pred_unexp))
            loss_exp = sum([criterion(pred_exp[i].type(torch.float64), exp_dec_value[i]) for i in range(n_batch)])
            loss_unexp = sum(
                [criterion(pred_unexp[i], torch.zeros_like(pred_unexp[i]).to(device)) for i in range(n_batch)])

        else:
            loss_exp = sum(
                [criterion(pred_exp[i][0], pred_exp[i][1], pred_exp[i][2], exp_dec_value[i]) for i in range(n_batch)])
            loss_unexp = sum([criterion(pred_unexp[i][0], pred_unexp[i][1], pred_unexp[i][2]) for i in range(n_batch)])

        # TODO: check the reg_loss and hyper-parameter

        reg_loss = (1 / 2) * sum([model.cell_feature[i].norm(2).pow(2) for i in range(n_batch)] + [
            model.gene_feature.norm(2).pow(2)]) / float(adata.n_obs + n_genes)

        loss = loss_exp + gamma * loss_unexp + 0.0001 * reg_loss

        if decoder == 'ZINB':
            ridge = sum(
                [torch.square(pred_exp[i][2]).mean() + torch.square(pred_unexp[i][2]).mean() for i in range(n_batch)])
            loss = recon_ratio * loss + 1e-3 * ridge
        else:
            loss = recon_ratio * loss

        loss = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     model.weights.data[0] = torch.clamp(model.weights.data[0], 0, 1)
        #     model.weights.data[1] = torch.clamp(model.weights.data[1], 0, 1)
        #     model.weights.data[2] = torch.clamp(model.weights.data[2], 0, 1)

        all_loss.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred_val_exp = model(exp_enc_graph, exp_val_graph, adj_list)
        model.train()

        if decoder in ['Dot', 'GMF']:
            # loss_val = criterion(pred_val_exp, exp_val_value)
            loss_val = sum([criterion(pred_val_exp[i].type(torch.float64), exp_val_value[i]) for i in range(n_batch)])
        else:
            loss_val = sum(
                [criterion(pred_val_exp[i][0], pred_val_exp[i][1], pred_val_exp[i][2], exp_val_value[i]) for i in
                 range(n_batch)])

        all_loss_val.append(loss_val.item())

        if (iter_idx + 1) % log_interval == 0:

            unexp_dec_graph = []
            for i in range(n_batch):
                unexp_sample_index = np.random.choice(all_unexp_index[i], n_neg_edges[i])
                unexp_sample = (unexp_edges[i][0][unexp_sample_index], unexp_edges[i][1][unexp_sample_index])
                unexp_dec_graph.append(
                    dgl.heterograph({('cell' + str(i + 1), 'exp' + str(i + 1), 'gene'): unexp_sample},
                                    num_nodes_dict={'cell' + str(i + 1): n_cells[i], 'gene': n_genes}).to(device))

        # adata.obsm['feat_'] = np.concatenate([model.emb_cell[i].data.cpu().numpy() for i in range(n_batch)], axis=0)
        # adata.varm['feat_'] = model.emb_gene.data.cpu().numpy()
    with torch.no_grad():
        adata.obsm['feat'] = np.concatenate([model.emb_cell[i].data.cpu().numpy() for i in range(n_batch)], axis=0)
        adata.varm['feat'] = model.emb_gene.data.cpu().numpy()
    if MNN == True:
        print('Train with MNN')
        iter_comb = None
        for epoch in tqdm(range(epoch_1, epoch_2)):
            if epoch % 100 == 0 or epoch == 500:
                if verbose:
                    print('Update spot triplets at epoch ' + str(epoch))
                adata.obsm['feat'] = np.concatenate([model.emb_cell[i].data.cpu().numpy() for i in range(n_batch)],
                                                    axis=0)
                adata.varm['feat'] = model.emb_gene.data.cpu().numpy()

                # If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
                # not all points have MNN achors
                mnn_dict = create_dictionary_mnn(adata, use_rep='feat', batch_name='batch', k=k,
                                                 iter_comb=iter_comb, verbose=0)

                anchor_ind = []
                positive_ind = []
                negative_ind = []
                for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                    batchname_list = adata.obs['batch'][mnn_dict[batch_pair].keys()]
                    #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
                    #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(section_ids)):
                        cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                            adata.obs['batch'] == section_ids[batch_id]].values

                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for anchor in mnn_dict[batch_pair].keys():
                        anchor_list.append(anchor)
                        ## np.random.choice(mnn_dict[batch_pair][anchor])
                        positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                        positive_list.append(positive_spot)
                        section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                        negative_list.append(
                            cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                    batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
                    anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                    positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                    negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

            model.train()
            optimizer.zero_grad()
            emb_list = [model.emb_cell[i].data for i in range(n_batch)]

            # 使用 torch.cat() 在维度0上连接张量
            z = torch.cat(emb_list, dim=0)

            if sample_rate != 1:
                exp_dec_value = []
                exp_dec_graph = []
                for i in range(n_batch):
                    exp_sample_index = np.random.choice(all_index[i], n_pos_edges[i])
                    exp_dec_value.append(exp_value[i][exp_sample_index])
                    exp_sample = (exp_graph[i].edges()[0][exp_sample_index], exp_graph[i].edges()[1][exp_sample_index])

                    exp_dec_graph.append(
                        dgl.heterograph({('cell' + str(i + 1), 'exp' + str(i + 1), 'gene'): exp_sample},
                                        num_nodes_dict={'cell' + str(i + 1): n_cells[i], 'gene': n_genes})
                        )

                    if decoder == 'ZINB':
                        exp_dec_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = \
                        exp_graph[i].nodes['cell' + str(i + 1)].data['sz_factor']
                        exp_dec_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']

            if decoder == 'ZINB':
                for i in range(n_batch):
                    unexp_dec_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = \
                    exp_graph[i].nodes['cell' + str(i + 1)].data['sz_factor']
                    unexp_dec_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']
                    exp_val_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = \
                    exp_graph[i].nodes['cell' + str(i + 1)].data['sz_factor']
                    exp_val_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']

            pred_exp, pred_unexp = model(exp_enc_graph, exp_dec_graph, adj_list, unexp_dec_graph)

            if decoder in ['Dot', 'GMF']:
                # loss_exp = criterion(pred_exp, exp_dec_value)
                # loss_unexp = criterion(pred_unexp, torch.zeros_like(pred_unexp))
                loss_exp = sum([criterion(pred_exp[i].type(torch.float64), exp_dec_value[i]) for i in range(n_batch)])
                loss_unexp = sum(
                    [criterion(pred_unexp[i], torch.zeros_like(pred_unexp[i]).to(device)) for i in range(n_batch)])

            else:
                loss_exp = sum([criterion(pred_exp[i][0], pred_exp[i][1], pred_exp[i][2], exp_dec_value[i]) for i in
                                range(n_batch)])
                loss_unexp = sum(
                    [criterion(pred_unexp[i][0], pred_unexp[i][1], pred_unexp[i][2]) for i in range(n_batch)])

            # TODO: check the reg_loss and hyper-parameter

            reg_loss = (1 / 2) * sum([model.cell_feature[i].norm(2).pow(2) for i in range(n_batch)] + [
                model.gene_feature.norm(2).pow(2)]) / float(adata.n_obs + n_genes)

            loss = loss_exp + gamma * loss_unexp + 0.0001 * reg_loss

            if decoder == 'ZINB':
                ridge = sum([torch.square(pred_exp[i][2]).mean() + torch.square(pred_unexp[i][2]).mean() for i in
                             range(n_batch)])
                loss = recon_ratio * loss + 1e-3 * ridge
            else:
                loss = recon_ratio * loss

            anchor_arr = z[anchor_ind,]
            positive_arr = z[positive_ind,]
            negative_arr = z[negative_ind,]

            triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

            loss = beta * loss + alpha * tri_output
            loss.backward()
            optimizer.step()

    # model.eval()
    # with torch.no_grad():
    # adata.obsm['no_batch'] = np.concatenate([model.emb_cell[i].data.cpu().numpy() for i in range(n_batch)], axis=0)
    # adata.varm['no_batch'] = model.emb_gene.data.cpu().numpy()

    del model
    # del all_exp
    gc.collect()
    torch.cuda.empty_cache()

    return adata




class SLGnet:
    def __init__(self,
                 adata,
                 cl_type=None,
                 n_clusters = 7,
                 gene_similarity = False,
                 alpha1 = 0.9,
                 n_layers = 2,
                 feats_dim = 64,
                 drop_out = 0.,
                 lr_1 = 0.01,
                 lr_2 = 0.01,
                 sample_rate = 0.1,
                 resolution = 1.0,
                 seed = 100,
                 use_rep = 'feat',
                 verbose = True,
                 return_all = False,
                 use_mclust = True,
                 ret = None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 neigh = 7,
                 dec_cluster_n = 7,
                 epochs_pre = 900,
                 n_epo = 900
                 ):
        self.seed = seed
        self.ret = False
        self.dec_cluster_n = dec_cluster_n
        self.epochs_pre = epochs_pre
        self.n_epo = n_epo
        self.cl_type = cl_type
        self.n_clusters = n_clusters
        self.gene_similarity = gene_similarity
        self.alpha1 = alpha1
        self.n_layers = n_layers
        self.feats_dim = feats_dim
        self.drop_out = drop_out
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.use_rep = use_rep
        self.verbose = verbose
        self.return_all = return_all
        self.use_mclust = use_mclust
        self.neigh = neigh
        self.device = device

        self.n_cells, self.n_genes = adata.X.shape
        self.graph_dict = graph_construction(adata, n=neigh)

        if 'mask' in self.graph_dict:
            mask = True
            adj_mask = self.graph_dict['mask'].to(self.device)
        else:
            mask = False
        setup_seed(self.seed)
        self.adj_norm = self.graph_dict["adj_norm"].to(self.device)
        self.adj_label = self.graph_dict["adj_label"].to(self.device)
        self.norm_value = self.graph_dict["norm_value"]

        assert self.sample_rate <= 1, "Please set 0<sample_rate<=1"

        ####################   Prepare data for training   ####################
        self.cell_type = adata.obs[self.cl_type].values if self.cl_type else None
        raw_exp =  False

        if self.cell_type is not None:
            self.n_clusters = self.n_clusters

        self.adata, exp_value, enc_graph, self.exp_dec_graph_, self.unexp_edges, self.coexp_edges, self.uncoexp_edges = make_graph(adata, raw_exp = False,
                                                                                                          gene_similarity = False)
        self.enc_graph, self.exp_value = enc_graph.to(self.device), torch.tensor(exp_value, device=self.device)
        self.model = lightST(p_drop=0.2,
                             alpha2=1.0,
                             dec_cluster_n = self.dec_cluster_n,
                             n_layers=self.n_layers,
                             n_cells=self.n_cells,
                             n_genes=self.n_genes,
                             drop_out=self.drop_out,
                             alpha1=self.alpha1,
                             learnable_weight=False,
                             gene_similarity=self.gene_similarity,
                             feats_dim=self.feats_dim).to(self.device)

        self.n_pos_edges, self.n_neg_edges = int(self.sample_rate * len(self.exp_value)), int(self.sample_rate * len(self.exp_value))
        self.n_neg_genes = len(self.coexp_edges[0]) if self.gene_similarity else None

        self.all_exp_index, self.all_unexp_index = np.arange(len(self.exp_value)), np.arange(len(self.unexp_edges[0]))
        self.all_uncoexp_index = np.arange(len(self.uncoexp_edges[0])) if self.gene_similarity else None

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def train(
            self,
            decay = 0.0,
            N = 1,
    ):

        epochs = self.epochs_pre
        lr = self.lr_1
        optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr,
            weight_decay=decay)

        self.model.train()
        criterion = nn.MSELoss()
        gene_cls = nn.BCELoss() if self.gene_similarity else None
        if self.sample_rate == 1:
            self.pos_graph, self.pos_value = self.exp_dec_graph_.to(self.device), self.exp_value
        for iter_idx in tqdm(range(epochs)):
            # Sample un-expressed / un-co-expressed edges, construct negative graph
            neg_edges = {}

            unexp_sample_index = np.random.choice(self.all_unexp_index, self.n_neg_edges)
            neg_edges[('cell', 'exp', 'gene')] = (
            self.unexp_edges[0][unexp_sample_index], self.unexp_edges[1][unexp_sample_index])
            if self.gene_similarity:
                uncoexp_sample_index = np.random.choice(self.all_uncoexp_index, self.n_neg_genes)
                neg_edges[('gene', 'co-exp', 'gene')] = (
                    self.uncoexp_edges[0][uncoexp_sample_index], self.uncoexp_edges[1][uncoexp_sample_index])

            neg_graph = dgl.heterograph(neg_edges, num_nodes_dict={'cell': self.n_cells, 'gene': self.n_genes}).to(self.device)
            # Add cell/gene size factor to negative graph

            # Sample expressed, construct positive graph
            if self.sample_rate != 1:
                pos_edges = {}

                exp_sample_index = np.random.choice(self.all_exp_index, self.n_pos_edges)
                pos_value = self.exp_value[exp_sample_index]
                exp_dec_edges = self.exp_dec_graph_[('cell', 'exp', 'gene')].edges()
                pos_edges[('cell', 'exp', 'gene')] = (
                    exp_dec_edges[0][exp_sample_index], exp_dec_edges[1][exp_sample_index])
                if self.gene_similarity: pos_edges[('gene', 'co-exp', 'gene')] = self.coexp_edges

                pos_graph = dgl.heterograph(pos_edges, num_nodes_dict={'cell': self.n_cells, 'gene': self.n_genes}).to(self.device)

            # Feed forward
            pos_pre, neg_pre, _ = self.model(self.enc_graph, pos_graph, self.adj_norm, neg_graph)
            # Calculate loss for regularization
            reg_loss = (1 / 2) * (self.model.cell_feature.norm(2).pow(2) +
                                  self.model.gene_feature.norm(2).pow(2)) / float(self.n_cells + self.n_genes)
            pos_value = pos_value.float()
            # Calculate loss for (un)expression

            loss_exp = criterion(pos_pre, pos_value)
            loss_unexp = criterion(neg_pre, torch.zeros_like(neg_pre))
            ridge = torch.square(pos_pre[2]).mean() + torch.square(neg_pre[2]).mean()
            reg_loss = reg_loss + 1e-3 * ridge

            loss = loss_exp + loss_unexp + 0.0001 * reg_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def process(self):
        self.model.eval()
        with torch.no_grad():
            c_feat, g_feat, c_last, g_last, q = self.model.encode(self.enc_graph, self.adj_norm)

        self.adata.obsm['feat'] = c_feat.cpu().detach().numpy()  # Return the weighted cell embeddings

        adata = kmeans(self.adata, n_clusters=self.n_clusters, use_rep=self.use_rep)
        y_pred_k = np.array(adata.obs['kmeans'])
        #nmi_k, ari_k = calculate_metric(adata.obs['ground_truth'], y_pred_k)
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        adata.obsm['feat'] = c_feat.cpu().numpy()  # Return the weighted cell embeddings
        adata.varm['feat'] = g_feat.detach().cpu().numpy()  # Return the final layer's gene embeddings
        adata.obs["pred"] = y_pred
        adata.obs["pred"] = adata.obs["pred"].astype('category')
        y_pred_q = adata.obs["pred"]
        #nmi_q, ari_q = calculate_metric(adata.obs['ground_truth'], y_pred_q)
        if self.ret == False:
            return None
        else:
            return c_feat, g_feat, q

    def DEC(
        self,
        dec_interval = 20,
        dec_tol = 0.00,
        N = 1,

    ):
        lr_2 = self.lr_2
        epochs = self.n_epo
        optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr_2,
            weight_decay=0)
        self.train()
        Kmeans = KMeans(n_clusters=self.dec_cluster_n, n_init=20, random_state=42)
        c_feat, g_feat, _ = self.process()
        #print(ari_)
        y_pred_last = np.copy(Kmeans.fit_predict(c_feat.cpu()))
        self.model.cluster_layer.data = torch.tensor(Kmeans.cluster_centers_).to(self.device)
        self.model.train()

        for epoch_id in tqdm(range(epochs)):
            # DEC clustering update
            if epoch_id % dec_interval == 0:
                _, _, tmp_q = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            neg_edges = {}

            unexp_sample_index = np.random.choice(self.all_unexp_index, self.n_neg_edges)
            neg_edges[('cell', 'exp', 'gene')] = (
            self.unexp_edges[0][unexp_sample_index], self.unexp_edges[1][unexp_sample_index])
            if self.gene_similarity:
                uncoexp_sample_index = np.random.choice(self.all_uncoexp_index, self.n_neg_genes)
                neg_edges[('gene', 'co-exp', 'gene')] = (
                    self.uncoexp_edges[0][uncoexp_sample_index], self.uncoexp_edges[1][uncoexp_sample_index])

            neg_graph = dgl.heterograph(neg_edges, num_nodes_dict={'cell': self.n_cells, 'gene': self.n_genes}).to(self.device)
            # Add cell/gene size factor to negative graph

            # Sample expressed, construct positive graph
            if self.sample_rate != 1:
                pos_edges = {}

                exp_sample_index = np.random.choice(self.all_exp_index, self.n_pos_edges)
                pos_value = self.exp_value[exp_sample_index]
                exp_dec_edges = self.exp_dec_graph_[('cell', 'exp', 'gene')].edges()
                pos_edges[('cell', 'exp', 'gene')] = (
                    exp_dec_edges[0][exp_sample_index], exp_dec_edges[1][exp_sample_index])
                if self.gene_similarity: pos_edges[('gene', 'co-exp', 'gene')] = self.coexp_edges

                pos_graph = dgl.heterograph(pos_edges, num_nodes_dict={'cell': self.n_cells, 'gene': self.n_genes}).to(self.device)

            # Feed forward
            pos_pre, neg_pre, out_q = self.model(self.enc_graph, pos_graph, self.adj_norm, neg_graph)
            loss = self.loss_function( torch.tensor(tmp_p).to(self.device), out_q).to(self.device)
            loss.backward()
            optimizer.step()

    def train_with_dec(
            self,
            dec_interval=20,
            dec_tol=0.00,
            N=1,
    ):
        lr_2 = self.lr_2
        epochs = self.n_epo
        optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr_2,
            weight_decay=0)
        self.train()
        Kmeans = KMeans(n_clusters=self.dec_cluster_n, n_init=20, random_state=42)
        c_feat, g_feat, _  = self.process()
        #print(ari_)
        y_pred_last = np.copy(Kmeans.fit_predict(c_feat.cpu()))

        self.model.cluster_layer.data = torch.tensor(Kmeans.cluster_centers_).to(self.device)
        self.model.train()
        criterion = nn.MSELoss()
        gene_cls = nn.BCELoss() if self.gene_similarity else None
        if self.sample_rate == 1:
            self.pos_graph, self.pos_value = self.exp_dec_graph_.to(self.device), self.exp_value
        for epoch_id in tqdm(range(epochs)):
            # Sample un-expressed / un-co-expressed edges, construct negative graph
            if epoch_id % dec_interval == 0:
                _, _, tmp_q  = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            neg_edges = {}

            unexp_sample_index = np.random.choice(self.all_unexp_index, self.n_neg_edges)
            neg_edges[('cell', 'exp', 'gene')] = (
            self.unexp_edges[0][unexp_sample_index], self.unexp_edges[1][unexp_sample_index])
            if self.gene_similarity:
                uncoexp_sample_index = np.random.choice(self.all_uncoexp_index, self.n_neg_genes)
                neg_edges[('gene', 'co-exp', 'gene')] = (
                    self.uncoexp_edges[0][uncoexp_sample_index], self.uncoexp_edges[1][uncoexp_sample_index])

            neg_graph = dgl.heterograph(neg_edges, num_nodes_dict={'cell': self.n_cells, 'gene': self.n_genes}).to(self.device)
            # Add cell/gene size factor to negative graph

            # Sample expressed, construct positive graph
            if self.sample_rate != 1:
                pos_edges = {}

                exp_sample_index = np.random.choice(self.all_exp_index, self.n_pos_edges)
                pos_value = self.exp_value[exp_sample_index]
                exp_dec_edges = self.exp_dec_graph_[('cell', 'exp', 'gene')].edges()
                pos_edges[('cell', 'exp', 'gene')] = (
                    exp_dec_edges[0][exp_sample_index], exp_dec_edges[1][exp_sample_index])
                if self.gene_similarity: pos_edges[('gene', 'co-exp', 'gene')] = self.coexp_edges

                pos_graph = dgl.heterograph(pos_edges, num_nodes_dict={'cell': self.n_cells, 'gene': self.n_genes}).to(self.device)

            # Feed forward
            pos_pre, neg_pre, out_q = self.model(self.enc_graph, pos_graph, self.adj_norm, neg_graph)
            # Calculate loss for regularization
            reg_loss = (1 / 2) * (self.model.cell_feature.norm(2).pow(2) +
                                  self.model.gene_feature.norm(2).pow(2)) / float(self.n_cells + self.n_genes)
            pos_value = pos_value.float()
            # Calculate loss for (un)expression
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            loss_exp = criterion(pos_pre, pos_value)
            loss_unexp = criterion(neg_pre, torch.zeros_like(neg_pre))
            ridge = torch.square(pos_pre[2]).mean() + torch.square(neg_pre[2]).mean()
            reg_loss = reg_loss + 1e-3 * ridge

            loss = loss_exp + loss_unexp +0.1*loss_kl + 0.0001 * reg_loss


            loss.backward()
            optimizer.step()