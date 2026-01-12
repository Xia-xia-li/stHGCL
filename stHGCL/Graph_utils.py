import dgl
import numpy as np
import pandas as pd
import torch
import scipy as sp
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


#############################################heterogeneous graph################################################

def identity_mapping(x):
    return x

activation_map = {
    'relu' : F.relu,
    'leaky' : F.leaky_relu,
    'selu' : F.selu,
    'sigmoid' : F.sigmoid,
    'tanh' : F.tanh,
    'none' : identity_mapping
}

def sparse_to_torch(sp,device):
    coo = sp.tocoo()
    values = torch.FloatTensor(coo.data)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    shape = coo.shape

    return torch.sparse.FloatTensor(indices, values, torch.Size(shape),)

def get_degree_inv(graph, edge_types, reverse):
    if reverse:
        degree = sum(graph.in_degrees(etype = str(etype)) for etype in edge_types)
    else:
        degree = sum(graph.out_degrees(etype = str(etype)) for etype in edge_types)

    degree_inv = torch.pow(degree, -0.5)
    degree_inv[torch.isinf(degree_inv)] = 0.
    # D_inv = torch.diag(degree_inv)#sp.diags(degree_inv.cpu().numpy())
    n_dots = len(degree_inv)
    idx = [list(range(n_dots)),list(range(n_dots))]

    return torch.sparse_coo_tensor(idx, degree_inv, (n_dots, n_dots))


def get_adjacency(graph, edge_types, ckey = 'cell', gkey = 'gene'):
    n_cells, n_genes = graph.num_nodes(ckey), graph.num_nodes(gkey)

    adj_uv = torch.sparse_coo_tensor(size = (n_cells, n_genes))
    for etype in edge_types:
        adj_uv += graph.adjacency_matrix(etype = str(etype))

    return adj_uv.to(graph.device)

def degree_noramlization(graph, edge_types, ckey = 'cell', gkey = 'gene'):
    adj_uv = get_adjacency(graph, edge_types, ckey, gkey)
    degree_u = get_degree_inv(graph, edge_types, False)
    degree_v = get_degree_inv(graph, edge_types, True)

    normed_adj_u = torch.sparse.mm(degree_u, adj_uv)
    normed_adj_u = torch.sparse.mm(normed_adj_u, degree_v)

    normed_adj_v = torch.sparse.mm(degree_v, adj_uv.t())
    normed_adj_v = torch.sparse.mm(normed_adj_v, degree_u)

    return normed_adj_u, normed_adj_v

def add_degree(graph, edge_types):
    def _calc_norm(x):
        x = x.numpy().astype('float32')
        x[x == 0.] = np.inf
        x = torch.FloatTensor(1. / np.sqrt(x))
        return x.unsqueeze(1)

    cell_ci, gene_ci = _calc_norm(graph['reverse-exp'].in_degrees()), _calc_norm(graph['exp'].in_degrees())
    cell_cj, gene_cj = _calc_norm(graph['exp'].out_degrees()), _calc_norm(graph['reverse-exp'].out_degrees())
    graph.nodes['cell'].data.update({'ci': cell_ci, 'cj': cell_cj})
    graph.nodes['gene'].data.update({'ci': gene_ci, 'cj': gene_cj})

    if 'co-exp' in edge_types:
        gene_cii, gene_cjj = _calc_norm(graph['co-exp'].in_degrees()), _calc_norm(graph['co-exp'].out_degrees())
        graph.nodes['gene'].data.update({'cii': gene_cii, 'cjj': gene_cjj})

def add_biggraph_degree(graph, edge_types, symmetric = True, n_cells = None, n_genes = None):
    def _calc_norm(x):
        x = x.numpy().astype('float32')
        x[x == 0.] = np.inf
        x = torch.FloatTensor(1. / np.sqrt(x))

        return x.unsqueeze(1)

    gene_ci = []
    gene_cj = []

    for i in range(len(edge_types)):
        cell_ci = []
        cell_cj = []

        cell_ci.append(graph['reverse-exp'+str(i+1)].in_degrees())
        cell_cj.append(graph['exp'+str(i+1)].out_degrees())

        gene_ci.append(graph[f'{edge_types[i]}'].in_degrees())
        if symmetric:
            gene_cj.append(graph[f'reverse-{edge_types[i]}'].out_degrees())

        cell_ci = _calc_norm(sum(cell_ci))

        if symmetric:
            cell_cj = _calc_norm(sum(cell_cj))

        graph.nodes['cell'+str(i+1)].data.update({'ci': cell_ci, 'cj': cell_cj})


    gene_ci = _calc_norm(sum(gene_ci))
    if symmetric:
        gene_cj = _calc_norm(sum(gene_cj))

    graph.nodes['gene'].data.update({'ci': gene_ci, 'cj': gene_cj})

def make_graph(adata, raw_exp=False, gene_similarity=False):
    X = adata.X
    num_cells, num_genes = X.shape

    # Make expressioin/train graph
    num_nodes_dict = {'cell': num_cells, 'gene': num_genes}
    exp_train_cell, exp_train_gene = np.where(X > 0)
    unexp_edges = np.where(X == 0)

    # expression edges
    exp_edge_dict = {
        ('cell', 'exp', 'gene'): (exp_train_cell, exp_train_gene),
        ('gene', 'reverse-exp', 'cell'): (exp_train_gene, exp_train_cell)
    }

    coexp_edges, uncoexp_edges = None, None
    if gene_similarity:
        coexp_edges, uncoexp_edges = construct_gene_graph(X)
        exp_edge_dict[('gene', 'co-exp', 'gene')] = coexp_edges

    # expression encoder/decoder graph
    enc_graph = dgl.heterograph(exp_edge_dict, num_nodes_dict=num_nodes_dict)

    exp_edge_dict.pop(('gene', 'reverse-exp', 'cell'))
    dec_graph = dgl.heterograph(exp_edge_dict, num_nodes_dict=num_nodes_dict)

    # add degree to cell/gene nodes
    add_degree(enc_graph, ['exp'] + (['co-exp'] if gene_similarity else []))

    # If use ZINB decoder, add size factor to cell/gene nodes
    if raw_exp:
        Raw = pd.DataFrame(adata.raw.X, index=list(adata.raw.obs_names), columns=list(adata.raw.var_names))
        X = Raw[list(adata.var_names)].values
        exp_value = X[exp_train_cell, exp_train_gene].reshape(-1, 1)
        dec_graph.nodes['cell'].data['cs_factor'] = torch.Tensor(adata.obs['cs_factor']).reshape(-1, 1)
        dec_graph.nodes['gene'].data['gs_factor'] = torch.Tensor(adata.var['gs_factor']).reshape(-1, 1)

    else:
        ## Deflate the edge values of the bipartite graph to between 0 and 1
        X = X / adata.var['gs_factor'].values
        exp_value = X[exp_train_cell, exp_train_gene].reshape(-1, 1)

    return adata, exp_value, enc_graph, dec_graph, unexp_edges, coexp_edges, uncoexp_edges

def make_big_graph(single_adata,n_batch, raw_exp=False, highly_variable=None, val_ratio=0.02):

    num_nodes_dict = {}
    exp_train_dict = {}
    exp_value = []
    exp_dec_graph = []
    exp_val_graph = []
    exp_val_value = []
    unexp_edges = []

    for i in range(n_batch):
        X = single_adata[i].X.toarray()
        num_cells, num_genes = X.shape
        num_nodes_dict.update({'cell'+str(i+1): num_cells, 'gene': num_genes})


        gene_factor = np.max(X, axis=0, keepdims=True)
        single_adata[i].var['gene_factor'] = gene_factor.reshape(-1)
        exp_cell, exp_gene = np.where(X > 0)
        unexp_edges.append(np.where(X == 0))

        idx = np.arange(len(exp_cell))  # 对存在的边的数量的编码
        np.random.shuffle(idx)  # 打乱边的顺序
        num_valid = int(np.ceil(len(exp_cell) * val_ratio))  # 从存在的边中挑选一定比例作为验证的边的数量
        idx_train, idx_valid = idx[num_valid:], idx[:num_valid]  ##分训练集和验证集索引(存在的边的)
        idx_train.sort()  # 索引升序排列
        idx_valid.sort()

        exp_valid_cell, exp_valid_gene = exp_cell[idx_valid], exp_gene[idx_valid]
        exp_train_cell, exp_train_gene = exp_cell[idx_train], exp_gene[idx_train]
        exp_train_dict.update({
            ('cell'+str(i+1), 'exp'+str(i+1), 'gene'): (exp_train_cell, exp_train_gene),
            ('gene', 'reverse-exp'+str(i+1), 'cell'+str(i+1)): (exp_train_gene, exp_train_cell)
                               })
        exp_dec_graph.append(
            dgl.heterograph({('cell'+str(i+1), 'exp'+str(i+1), 'gene'): (exp_train_cell, exp_train_gene)},
                                         num_nodes_dict={'cell'+str(i+1): num_cells, 'gene': num_genes}))
        exp_val_graph.append(
            dgl.heterograph({('cell'+str(i+1), 'exp'+str(i+1), 'gene'): (exp_valid_cell, exp_valid_gene)},
                                         num_nodes_dict={'cell'+str(i+1): num_cells, 'gene': num_genes}))

        if raw_exp:
            if highly_variable[0] == None:
                exp_value.append(single_adata[i].raw.X[exp_train_cell, exp_train_gene].reshape(-1, 1))
                exp_val_value.append(single_adata[i].raw.X[exp_valid_cell, exp_valid_gene].reshape(-1, 1))

                exp_dec_graph[i].nodes['cell'+str(i+1)].data['sz_factor'] = torch.Tensor(single_adata[i].obs['sz_factor']).reshape(-1, 1)
                exp_dec_graph[i].nodes['gene'].data['ge_factor'] = torch.Tensor(gene_factor).reshape(-1, 1)

            else:
                exp_value.append(single_adata[i].raw[:, highly_variable].X[exp_train_cell, exp_train_gene].reshape(-1, 1))
                exp_val_value.append(single_adata[i].raw[:, highly_variable].X[exp_valid_cell, exp_valid_gene].reshape(-1, 1))

                exp_dec_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = torch.Tensor(single_adata[i].obs['sz_factor']).reshape(-1, 1)
                exp_dec_graph[i].nodes['gene'].data['ge_factor'] = torch.Tensor(gene_factor).reshape(-1, 1)

        else:
            X = X / gene_factor  #使边表达值放缩到0-1之间
            exp_value.append(X[exp_train_cell, exp_train_gene].reshape(-1, 1))
            exp_val_value.append(X[exp_valid_cell, exp_valid_gene].reshape(-1, 1))

    exp_enc_graph = dgl.heterograph(exp_train_dict, num_nodes_dict=num_nodes_dict)
    add_biggraph_degree(exp_enc_graph, ['exp'+str(i+1) for i in range(n_batch)])



    return single_adata, exp_value, exp_enc_graph, exp_dec_graph, exp_val_graph, exp_val_value, unexp_edges

def construct_gene_graph(gex_features, corr_method='cosine', corr_threshold=0.9):
    """Generate nodes, edges and edge weights for dataset.

    Parameters
    ----------
    gex_features: anndata.AnnData
        Gene data, contains feature matrix (.X) and feature names (.var['feature_types']).

    Returns
    --------
    uu: list[int]
        Predecessor node id of each edge.
    vv: list[int]
        Successor node id of each edge.
    ee: list[float]
        Edge weight of each edge.
    """

    if corr_method == 'pearson':
        corr = np.abs(np.corrcoef(gex_features, rowvar=False))
    elif corr_method == 'cosine':
        corr = cosine_similarity(gex_features.T)

    row, col = np.diag_indices_from(corr)
    corr[row, col] = 0

    coexp_edges = np.where(abs(corr) > corr_threshold)
    uncoexp_edges = np.where(abs(corr) < 1 - corr_threshold)
    # neg_idx = np.random.choice(len(nuu), 10*len(uu))
    # nuu, nvv = nuu[neg_idx], nvv[neg_idx]

    return coexp_edges, uncoexp_edges

###############################################homogeneous graph####################################################

def generate_adj_mat(adata, include_self=False, n=6):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm['spatial'])

    # sample_name = list(adata.uns['spatial'].keys())[0]
    # scalefactors = adata.uns['spatial'][sample_name]['scalefactors']
    # adj_mat = dist <= scalefactors['fiducial_diameter_fullres'] * (n+0.2)
    # adj_mat = adj_mat.astype(int)

    # n_neighbors = np.argpartition(dist, n+1, axis=1)[:, :(n+1)]
    # adj_mat = np.zeros((len(adata), len(adata)))
    # for i in range(len(adata)):
    #     adj_mat[i, n_neighbors[i, :]] = 1

    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n + 1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_graph(adj):
    """Symmetrically normalize adjacency matrix."""
    adj_ = adj + sp.eye(adj.shape[0])
    adj = scipy.sparse.coo_matrix(adj_)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)  # 点集
    return sparse_mx_to_torch_sparse_tensor(adj)


def preprocess_graph1(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = scipy.sparse.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def mask_generator(adj_label, N=1):
    adj_label = adj_label.coalesce()
    idx = adj_label.indices()
    cell_num = adj_label.size()[0]

    list_non_neighbor = []
    for i in range(0, cell_num):
        neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
        n_selected = len(neighbor) * N

        # non neighbors
        total_idx = torch.range(0, cell_num - 1, dtype=torch.float32)
        non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
        indices = torch.randperm(len(non_neighbor), dtype=torch.float32)
        random_non_neighbor = indices[:n_selected]
        list_non_neighbor.append(random_non_neighbor)

    x = adj_label.indices()[0]
    y = torch.concat(list_non_neighbor)

    indices = torch.stack([x, y])
    indices = torch.concat([adj_label.indices(), indices], axis=1)

    value = torch.concat([adj_label.values(), torch.zeros(len(x), dtype=torch.float32)])
    adj_mask = torch.sparse_coo_tensor(indices, value)

    return adj_mask


def graph_computing(pos, n):
    from scipy.spatial import distance
    list_x = []
    list_y = []
    list_value = []

    for node_idx in range(len(pos)):
        tmp = pos[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, pos, 'euclidean')
        res = distMat.argsort()
        # tmpdist = distMat[0, res[0][1:params.k + 1]]
        for j in np.arange(1, n + 1):
            list_x += [node_idx, res[0][j]]
            list_y += [res[0][j], node_idx]
            list_value += [1, 1]

    adj = sp.csr_matrix((list_value, (list_x, list_y)))
    adj = adj >= 1
    adj = adj.astype(np.float32)
    return adj


def graph_construction(adata, n, dmax=50, mode='KNN'):
    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
        # adj_m1 = graph_computing(adata.obsm['spatial'], n=n)
    else:
        adj_m1 = generate_adj_mat(adata, dmax)
    adj_m1 = scipy.sparse.coo_matrix(adj_m1)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_m1 - scipy.sparse.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_norm_m1 = adj_norm_m1
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    # adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())

    adj_m1 = scipy.sparse.coo_matrix(adj_m1)
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)

    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    # # generate random mask
    adj_mask = mask_generator(adj_label_m1.to_sparse(), N=1)

    graph_dict = {
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1.coalesce(),
        "norm_value": norm_m1,
        "mask": adj_mask
    }

    return graph_dict
def generate_adj_list(Batch_list,n_batch,neigh,device):
    adj_list = []
    for i in range(n_batch):
        adata = Batch_list[i]
        graph_dict = graph_construction(adata, n=neigh)
        adj_list.append(graph_dict['adj_norm'].to(device))
    return adj_list
