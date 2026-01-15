import dgl
import torch
from torch import nn, optim
import torch.nn.functional as F
import dgl.function as fn


class LightGraphConv(nn.Module):
    def __init__(self,
                 drop_out=0.1):
        """Light Graph Convolution

        Paramters
        ---------
        drop_out : float
            dropout rate (neighborhood dropout)
        """
        super().__init__()
        self.dropout = nn.Dropout(drop_out)

    def forward(self, graph, feats):
        """Apply Light Graph Convoluiton to specific edge type {r}

        Paramters
        ---------
        graph : dgl.graph
        src_feats : torch.FloatTensor
            source node features

        ci : torch.LongTensor
            in-degree of sources ** (-1/2)
            shape : (n_sources, 1)
        cj : torch.LongTensor
            out-degree of destinations ** (-1/2)
            shape : (n_destinations, 1)

        Returns
        -------
        output : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{j -> i, r} = h_{j} / ( N_{i, r} * N_{j, r} )
                where N_{i, r} ; number of neighbors_{i, r} ** (1/2)
        2. aggregation
            \sum_{j \in N(i), r} MP_{j -> i, r}
        """
        if isinstance(feats, tuple):
            src_feats, dst_feats = feats

        with graph.local_scope():
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']

            cj_dropout = self.dropout(cj)
            weighted_feats = torch.mul(src_feats, cj_dropout)
            graph.srcdata['h'] = weighted_feats

            # graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'out'))
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'out'))
            out = torch.mul(graph.dstdata['out'], ci)

        return out


class LightGCNLayer(nn.Module):
    def __init__(self,
                 ckey,
                 drop_out=0.1):
        super().__init__()
        """LightGCN Layer

        edge_types : list
            all edge types
        drop_out : float
            dropout rate (feature dropout)
        """
        self.ckey = ckey
        self.n_batch = len(ckey)
        conv = {}

        for i in range(len(ckey)):
            cell_to_gene_key = 'exp' + str(i + 1)
            gene_to_cell_key = 'reverse-exp' + str(i + 1)
            conv[cell_to_gene_key] = LightGraphConv(drop_out=drop_out)
            conv[gene_to_cell_key] = LightGraphConv(drop_out=drop_out)

        self.conv = dgl.nn.HeteroGraphConv(conv, aggregate='mean')
        self.feature_dropout = nn.Dropout(drop_out)
        self.normed_adj_u, self.normed_adj_v = None, None

    def forward(self, graph, ufeats, ifeats, ckey=['cell1', 'cell2'], gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.graph
        ufeats, ifeats : torch.FloatTensor
            node features
        ckey, gkey : str
            target node types

        Returns
        -------
        ufeats, ifeats : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{i} = \{ MP_{i, r_{1}}, MP_{i, r_{2}}, ... \}
        2. aggregation
            h_{i} = \sigma_{j \in N(i) , r} MP_{i, j, r}
        """
        feats = {
            ckey[i]: ufeats[i]
            for i in range(self.n_batch)
        }
        feats[gkey] = ifeats

        out = self.conv(graph, feats)

        return [out[key] for key in ckey], out[gkey]
        # return out[ckey[0]], out[ckey[1]], out[ckey[2]], out[ckey[3]], out[gkey]


class GraphConv(nn.Module):
    def __init__(self, n_cells, drop_out=0., act=F.relu):
        super(GraphConv, self).__init__()
        self.n_cells = n_cells
        self.dropout = drop_out
        self.act = act

    def forward(self, adj, feats):
        feats = F.dropout(feats, self.dropout)
        out = torch.sparse.mm(adj, feats)  # Apply adjacency matrix
        return out


class LightGCN_multi(nn.Module):
    def __init__(self,
                 n_layers,
                 n_cells,
                 n_genes,
                 drop_out,
                 feats_dim,
                 adj_list,
                 decoder='Dot',
                 learnable_weight=False):
        super().__init__()
        """LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
        paper : https://arxiv.org/pdf/2002.02126.pdf

        n_layers : int
            number of GCMC layers
        edge_types : list
            all edge types
        drop_out : float
            dropout rate (neighbors)
        learnable_weight : boolean
            whether to learn weights for embedding aggregation
            if False, use 1/n_layers
        """

        self.n_cells = n_cells
        self.n_genes = n_genes
        self.n_batch = len(n_cells)
        self.ckey = ['cell{}'.format(i + 1) for i in range(self.n_batch)]
        self.gkey = 'gene'
        self.adj_list = adj_list
        # self.n_cells1 = n_cells[0]
        # self.n_cells2 = n_cells[1]
        # self.n_cells3 = n_cells[2]
        # self.n_cells4 = n_cells[3]

        self.gene_feature = nn.Parameter(torch.Tensor(self.n_genes, feats_dim))
        self.cell_feature = nn.ParameterList([nn.Parameter() for _ in range(self.n_batch)])
        self.batch = [None] * self.n_batch
        for i in range(len(n_cells)):
            self.cell_feature[i] = nn.Parameter(torch.Tensor(self.n_cells[i], feats_dim))
            batch_ = torch.zeros(self.n_cells[i], len(n_cells))
            batch_[:, i] = torch.ones(self.n_cells[i])
            self.batch[i] = batch_.to(torch.device('cuda'))

        # self.batch1 = torch.cat((torch.ones(self.n_cells1, 1).to(torch.device('cuda')), torch.zeros(self.n_cells1, 1).to(torch.device('cuda')), torch.zeros(self.n_cells1, 1).to(torch.device('cuda')), torch.zeros(self.n_cells1, 1).to(torch.device('cuda'))), dim=1)
        # self.batch2 = torch.cat((torch.zeros(self.n_cells2, 1).to(torch.device('cuda')), torch.ones(self.n_cells2, 1).to(torch.device('cuda')), torch.zeros(self.n_cells2, 1).to(torch.device('cuda')), torch.zeros(self.n_cells2, 1).to(torch.device('cuda'))), dim=1)
        # self.batch3 = torch.cat((torch.zeros(self.n_cells3, 1).to(torch.device('cuda')), torch.zeros(self.n_cells3, 1).to(torch.device('cuda')), torch.ones(self.n_cells3, 1).to(torch.device('cuda')), torch.zeros(self.n_cells3, 1).to(torch.device('cuda'))), dim=1)
        # self.batch4 = torch.cat((torch.zeros(self.n_cells4, 1).to(torch.device('cuda')), torch.zeros(self.n_cells4, 1).to(torch.device('cuda')), torch.zeros(self.n_cells4, 1).to(torch.device('cuda')), torch.ones(self.n_cells4, 1).to(torch.device('cuda')),), dim=1)

        # self.input_layer1 = nn.Sequential(nn.Linear(feats_dim+2, feats_dim), nn.BatchNorm1d(feats_dim), nn.ELU())
        # self.input_layer2 = nn.Sequential(nn.Linear(feats_dim+2, feats_dim), nn.BatchNorm1d(feats_dim), nn.ELU())
        self.emb_layer = nn.Sequential(nn.Linear(feats_dim + self.n_batch, feats_dim), nn.BatchNorm1d(feats_dim),
                                       nn.ELU())
        # self.emb_batch = nn.Sequential(nn.Linear(2, feats_dim), nn.BatchNorm1d(feats_dim), nn.ELU())
        # self.out_layer = nn.Sequential(nn.Linear(feats_dim, 32), nn.BatchNorm1d(32), nn.ELU())
        self.pred_pos = [None] * self.n_batch
        self.pred_neg = [None] * self.n_batch
        self.u_hidden = [None] * self.n_batch
        self.u_hidden_z = [None] * self.n_batch
        self.h_cell = [None] * self.n_batch

        # nn.init.xavier_uniform_(self.cell_feature) #均匀分布初始化
        # nn.init.xavier_uniform_(self.cell_feature2)
        # nn.init.xavier_uniform_(self.cell_feature3)
        # nn.init.xavier_uniform_(self.cell_feature4)
        nn.init.normal_(self.gene_feature, std=0.01)
        for i in range(self.n_batch):
            nn.init.normal_(self.cell_feature[i], std=0.01)

        self.n_layers = n_layers
        self.encoders = nn.ModuleList()
        self.Layer_block = []
        self.GNN_Layer = []

        for i in range(1, self.n_batch + 1):
            layer_function = lambda: nn.ModuleList()
            setattr(self, 'gcn_encoders_' + str(i), layer_function)
            self.Layer_block.append(layer_function)

        for i in range(0, self.n_batch):
            Layer_i = self.Layer_block[i]()
            for _ in range(10):
                Layer_i.append(GraphConv(self.n_cells, drop_out=drop_out, act=F.relu))
            self.GNN_Layer.append(Layer_i)

        for _ in range(10):
            self.encoders.append(LightGCNLayer(ckey=self.ckey, drop_out=drop_out))

        # self.weights = torch.ones(n_layers + 1) / (n_layers + 1)
        # self.weights = torch.tensor([1., 1./2, 1./2])
        if self.n_layers == 2:
            # self.weights = torch.tensor([1., 1./4, 1.])
            self.weights = torch.tensor([1. / 3, 1. / 3, 1. / 3])
        else:
            self.weights = torch.tensor([1., 1. / 2, 1., 1. / 2, 1.])

        if learnable_weight:
            self.weights = nn.Parameter(self.weights)

        if decoder == 'Dot':
            self.decoder = DotDecoder()
        elif decoder == 'GMF':
            self.decoder = GMFDecoder(feats_dim=feats_dim)
        elif decoder == 'ZINB':
            self.decoder = ZINBDecoder(feats_dim=feats_dim)

        for p, q in self.decoder.named_parameters():  ##初始化decoder参数
            if 'weight' in p:
                nn.init.kaiming_normal_(q)
            elif 'bias' in p:
                nn.init.constant_(q, 0)

    # def encode(self, graph, ufeats, ifeats, ckey, gkey):
    #     u_hidden, i_hidden = self.weights[0] * ufeats, self.weights[0] * ifeats
    #     for w, encoder in zip(self.weights[1:], self.encoders):
    #         ufeats, ifeats = encoder(graph, ufeats, ifeats, ckey, gkey)
    #         u_hidden = u_hidden + w * ufeats
    #         i_hidden = i_hidden + w * ifeats
    #
    #     return u_hidden, i_hidden

    def encode(self, graph, adj, ufeats, ifeats, ckey, gkey):
        i_hidden = self.weights[0] * ifeats
        for i in range(self.n_batch):
            self.u_hidden[i] = self.weights[0] * ufeats[i]

        for w, encoder in zip(self.weights[1:], self.encoders):
            ufeats, ifeats = encoder(graph, ufeats, ifeats, ckey, gkey)
            i_hidden = i_hidden + w * ifeats
            for i in range(self.n_batch):
                self.u_hidden[i] = self.u_hidden[i] + w * ufeats[i]

            for i in range(self.n_batch):
                self.u_hidden_z[i] = self.GNN_Layer[i][0](adj[i], self.u_hidden[i])
                self.u_hidden_z[i] = self.GNN_Layer[i][1](adj[i], self.u_hidden_z[i])

        return self.u_hidden_z, i_hidden
        # 连续更新的特征 ，一次更新的特征

    def decode(self, pos_graph, neg_graph, ufeats, ifeats, ckey, gkey):
        # ufeats1, ufeats2, ufeats3, ufeats4 = self.emb_layer(torch.cat((ufeats1, self.batch1), dim=1)), self.emb_layer(torch.cat((ufeats2, self.batch2), dim=1)), self.emb_layer(torch.cat((ufeats3, self.batch3), dim=1)), self.emb_layer(torch.cat((ufeats4, self.batch4), dim=1))
        # ufeats1, ufeats2 = ufeats1 + self.emb_batch(self.batch1), ufeats2 + self.emb_batch(self.batch2)
        for i in range(self.n_batch):
            self.pred_pos[i] = self.decoder(pos_graph[i], ufeats[i], ifeats, ckey[i], gkey)
            self.pred_neg[i] = self.decoder(neg_graph[i], ufeats[i], ifeats, ckey[i], gkey)
        # self.pred_pos[0] = self.decoder(pos_graph1, ufeats1, ifeats, ckey1, gkey)
        # self.pred_neg[0] = self.decoder(neg_graph1, ufeats1, ifeats, ckey1, gkey)
        # self.pred_pos[1] = self.decoder(pos_graph2, ufeats2, ifeats, ckey2, gkey)
        # self.pred_neg[1] = self.decoder(neg_graph2, ufeats2, ifeats, ckey2, gkey)
        # self.pred_pos[2] = self.decoder(pos_graph3, ufeats3, ifeats, ckey3, gkey)
        # self.pred_neg[2] = self.decoder(neg_graph3, ufeats3, ifeats, ckey3, gkey)
        # self.pred_pos[3] = self.decoder(pos_graph4, ufeats4, ifeats, ckey4, gkey)
        # self.pred_neg[3] = self.decoder(neg_graph4, ufeats4, ifeats, ckey4, gkey)

        return self.pred_pos, self.pred_neg

    def forward(self,
                enc_graph,
                pos_graph,
                adj,
                neg_graph=None
                ):
        """
        Parameters
        ----------
        enc_graph : dgl.graph
        dec_graph : dgl.homograph

        Notes
        -----
        1. LightGCN encoder
            1 ) message passing
                MP_{j -> i, r} = h_{j} / ( N_{i, r} * N_{j, r} )
            2 ) aggregation
                \sum_{j \in N(i), r} MP_{j -> i, r}

        2. final features
            cell_{i} = mean( h_{i, layerself.cell_feature = {Parameter: (943, 75)} Parameter containing:\ntensor([[ 0.0007, -0.0501,  0.0644,  ..., -0.0756,  0.0526, -0.0293],\n        [ 0.0743, -0.0693, -0.0382,  ..., -0.0612,  0.0300,  0.0068],\n        [-0.0341, -0.0038,  0.0670,  ..., -0.0470, -0.0631, -0.0403],\n        ...,\n        [-0… View_1}, h_{i, layer_2}, ... )
            gene_{j} = mean( h_{j, layer_1}, h_{j, layer_2}, ... )

        3. Bilinear decoder
            logits_{i, j, r} = ufeats_{i} @ Q_r @ ifeats_{j}
        """

        # ufeats, ifeats = self.encode(enc_graph, self.cell_feature, self.gene_feature, ckey, gkey)
        # ufeats, ifeats, h_cell, h_gene = self.encode(enc_graph, self.cell_feature, self.gene_feature, self.ckey, self.gkey)
        ufeats, ifeats = self.encode(enc_graph, adj, self.cell_feature, self.gene_feature, self.ckey, self.gkey)
        # ufeats1, ufeats2 = self.out_layer(ufeats1), self.out_layer(ufeats2)
        self.emb_cell = ufeats
        self.emb_gene = ifeats
        # self.emb_cell1, self.emb_cell2 = h_cell1, h_cell2

        ####下面不能写出ufeats[0],ufeats[1],..= .., ..,...的形式
        # ufeats = self.emb_layer(torch.cat((ufeats[0], self.batch[0]), dim=1)), self.emb_layer(
        #     torch.cat((ufeats[1], self.batch[1]), dim=1)), self.emb_layer(
        #     torch.cat((ufeats[2], self.batch[2]), dim=1)), self.emb_layer(torch.cat((ufeats[3], self.batch[3]), dim=1))

        ###下面这步尝试修改为外面的for循环格式################################################################################################
        ufeats = [self.emb_layer(torch.cat((ufeats[i], self.batch[i]), dim=1)) for i in range(self.n_batch)]
        # for i in range(self.n_batch):
        #     ufeats[i] = self.emb_layer(torch.cat((ufeats[i], self.batch[i]), dim=1))

        # ufeats1, ufeats2 = ufeats1 + self.emb_batch(self.batch1), ufeats2 + self.emb_batch(self.batch2)

        if neg_graph:
            pred_pos, pred_neg = self.decode(pos_graph, neg_graph, ufeats, ifeats, self.ckey, self.gkey)
            return pred_pos, pred_neg
        else:
            for i in range(self.n_batch):
                self.pred_pos[i] = self.decoder(pos_graph[i], ufeats[i], ifeats, self.ckey[i], self.gkey)

        return self.pred_pos


class DotDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        """Dotproduct decoder for link prediction
        predict link existence (not edge type)
        """
        self.act = nn.Sigmoid()

    def forward(self, graph, ufeats, ifeats, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        ufeats : torch.FloatTensor
            cell features
        ifeats : torch.FloatTensor
            gene features

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_cells, 1)
        """

        with graph.local_scope():
            graph.nodes[ckey].data['h'] = ufeats
            graph.nodes[gkey].data['h'] = ifeats
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pred = self.act(graph.edata['score'])

        return pred

