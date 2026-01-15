import dgl
import numpy as np
import scanpy as sc
import torch
from torch import nn, optim
import torch.nn.functional as F
import gc
import dgl.function as fn
from .utils import setup_seed,mclust_R,calculate_metric,kmeans
from .Graph_utils import make_graph,graph_construction
from tqdm import tqdm


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

class DotDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        """Dotproduct decoder for link prediction
        predict link existence (not edge type)
        """
        self.act = nn.Sigmoid()

    def forward(self, graph, c_feat, g_feat, g_last=None, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        c_feat : torch.FloatTensor
            cell features
        g_feat : torch.FloatTensor
            gene features
        g_last : torch.FloatTensor
            gene features of the last layer
        ckey, gkey : str
            target node types

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_edges, 1)
        """

        with graph.local_scope():
            graph.nodes[ckey].data['h'] = c_feat
            graph.nodes[gkey].data['h'] = g_feat
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pred = self.act(graph.edata['score'])

        return pred

class GMFDecoder(nn.Module):
    def __init__(self, feats_dim):
        super().__init__()
        """Dotproduct decoder for link prediction
        predict link existence (not edge type)
        """
        self.out = nn.Sequential(nn.ReLU(inplace=True),nn.Linear(feats_dim, feats_dim),nn.ReLU(inplace=True),nn.Linear(feats_dim, 1), nn.Sigmoid())

    def forward(self, graph, c_feat, g_feat, g_last=None, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        c_feat : torch.FloatTensor
            cell features
        g_feat : torch.FloatTensor
            gene features
        g_last : torch.FloatTensor
            gene features of the last layer
        ckey, gkey : str
            target node types

        Returns
        -------
        pred : torch.FloatTensor
            shape : (n_edges, 1)
        """

        with graph.local_scope():
            graph.nodes[ckey].data['h'] = c_feat
            graph.nodes[gkey].data['h'] = g_feat
            graph.apply_edges(fn.u_mul_v('h', 'h', 'score'))
            pred = self.out(graph.edata['score'])

        return pred

class GraphConv(nn.Module):
    def __init__(self, n_cells , drop_out=0. , act=F.relu):
        super(GraphConv, self).__init__()
        self.n_cells = n_cells
        self.dropout = drop_out
        self.act = act
    def forward(self, adj,feats):
        feats = F.dropout(feats, self.dropout)
        out = torch.sparse.mm(adj, feats)  # Apply adjacency matrix
        return out


class LightGraphConv(nn.Module):
    def __init__(self, drop_out=0.1, gene2gene=False):
        super().__init__()
        self.dropout = nn.Dropout(drop_out)
        self.gene2gene = gene2gene

    def forward(self, graph, feats):
        if isinstance(feats, tuple):
            src_feats, dst_feats = feats

        with graph.local_scope():
            if self.gene2gene:
                cj, ci = graph.srcdata['cjj'], graph.dstdata['cii']
            else:
                cj, ci = graph.srcdata['cj'], graph.dstdata['ci']

            cj_dropout = self.dropout(cj)
            weighted_feats = torch.mul(src_feats, cj_dropout)
            graph.srcdata['h'] = weighted_feats

            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'out'))
            out = torch.mul(graph.dstdata['out'], ci)

        return out


class lightgraphconvlayer(nn.Module):
    def __init__(self, drop_out=0.1, alpha1=None):
        super().__init__()
        """lightgraphconv layer

        drop_out : float
            dropout rate (feature dropout)
        alpha1: float
            weight for gene massage
        """
        self.alpha1 = alpha1
        conv = {}

        cell_to_gene_key = 'exp'
        gene_to_cell_key = 'reverse-exp'
        gene_to_gene_key = 'co-exp'

        # convolution on cell -> gene graph
        conv[cell_to_gene_key] = LightGraphConv(drop_out=drop_out)

        # convolution on gene -> cell graph
        conv[gene_to_cell_key] = LightGraphConv(drop_out=drop_out)

        # convolution on gene -> gene graph
        if self.alpha1 is not None:
            conv[gene_to_gene_key] = LightGraphConv(drop_out=drop_out, gene2gene=True)

        self.conv = dgl.nn.HeteroGraphConv(conv, aggregate='stack')
        self.feature_dropout = nn.Dropout(drop_out)

    def forward(self, graph, c_feat, g_feat, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.graph
        c_feat, g_feat : torch.FloatTensor
            node features
        ckey, gkey : str
            target node types

        Returns
        -------
        c_feat, g_feat : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{i} = \{ MP_{i, r_{1}}, MP_{i, r_{2}}, ... \}
        2. aggregation
            h_{i} = \sigma_{j \in N(i) , r} MP_{i, j, r}
        """
        feats = {
            ckey: c_feat,
            gkey: g_feat
        }

        out = self.conv(graph, feats)
        c_feat = out[ckey].squeeze()
        g_feat = self.alpha1 * out[gkey][:, 0] + (1 - self.alpha1) * out[gkey][:, 1] if self.alpha1 is not None else out[gkey].squeeze()

        return c_feat, g_feat



class Graph_Comb(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        return output

class lightST(nn.Module):
    def __init__(self,
                 n_layers,
                 n_cells,
                 n_genes,
                 feats_dim=64,
                 drop_out=0.2,
                 learnable_weight=False,
                 gene_similarity=False,
                 alpha1=0.9,
                 p_drop=0.2,
                 alpha = 1.0,
                 alpha2=1.0,
                 dec_cluster_n = 7,
                 ):
        super().__init__()

        self.feats_dim = feats_dim
        self.p_drop = p_drop
        self.alpha2 = alpha2
        self.feats_dim = feats_dim
        self.dec_cluster_n = dec_cluster_n
        self.cluster_layer = nn.Parameter(torch.Tensor(self.dec_cluster_n, self.feats_dim))
        self.alpha = alpha
        # feature autoencoder
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # DEC cluster layer



        self.gene_similarity = gene_similarity
        self.alpha1 = alpha1 if gene_similarity else None

        self.n_cells = n_cells
        self.n_genes = n_genes

        self.cell_feature = nn.Parameter(torch.Tensor(self.n_cells, feats_dim))
        self.gene_feature = nn.Parameter(torch.Tensor(self.n_genes, feats_dim))


        #nn.init.xavier_uniform_(self.cell_feature)
        #nn.init.xavier_uniform_(self.gene_feature)
        nn.init.normal_(self.cell_feature, std=0.01)
        nn.init.normal_(self.gene_feature, std=0.01)


        self.n_layers = n_layers
        self.encoders = nn.ModuleList()
        self.gcn_encoders = nn.ModuleList()
        self.comb_encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(lightgraphconvlayer(drop_out=drop_out, alpha1=self.alpha1))
            self.gcn_encoders.append(GraphConv(self.n_cells,drop_out=drop_out,act=F.relu))
            self.comb_encoders.append(Graph_Comb(self.feats_dim))

        if self.n_layers == 2:
            self.weights = torch.tensor([1./3, 1. / 3, 1. / 3]).cuda()
        else:
            self.weights = torch.ones([self.n_layers + 1, 1]) / (self.n_layers + 1)

        if learnable_weight:
            self.weights = nn.Parameter(self.weights)


        self.decoder = DotDecoder()


        for p, q in self.decoder.named_parameters():
            if 'weight' in p:
                nn.init.kaiming_normal_(q)
            elif 'bias' in p:
                nn.init.constant_(q, 0)

    def encode(self, graph, adj, ckey='cell', gkey='gene'):
        w = self.weights[1:]
        c_feat, g_feat = self.cell_feature, self.gene_feature

        c_hidden, g_hidden = self.weights[0] * c_feat , self.weights[0] * g_feat

        c_feat_1, g_feat_1 = self.encoders[0](graph, c_feat, g_feat, ckey, gkey)
        #c_feat_1 = self.comb_encoders[0](c_feat1,c_feat1_)


        c_feat_2, g_feat_2 = self.encoders[1](graph, c_feat_1, g_feat_1, ckey, gkey)

        c_hidden_1 = c_hidden + w[0]*c_feat_1 + w[1]*c_feat_2
        c_hidden_2 = self.gcn_encoders[0](adj, c_hidden_1)
        c_hidden_z = self.gcn_encoders[1](adj, c_hidden_2)
        g_hidden_z = g_hidden + w[0]*g_feat_1 + w[1]*g_feat_2

        q = 1.0 / (1.0 + torch.sum(torch.pow(c_hidden_z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return c_hidden_z, g_hidden_z, c_feat, g_feat, q




    def decode(self, pos_graph, neg_graph, c_feat, g_feat, g_last, ckey, gkey):
        pos_pre = self.decoder(pos_graph, c_feat, g_feat, g_last, ckey, gkey)
        neg_pre = self.decoder(neg_graph, c_feat, g_feat, g_last, ckey, gkey)
        return pos_pre, neg_pre




    def forward(self,
                enc_graph,
                pos_graph,
                adj,
                neg_graph=None,
                ckey='cell',
                gkey='gene'):
        """
        Parameters
        ----------
        enc_graph, pos_graph, neg_graph: dgl.graph
            constructed graphs

        ckey, gkey : str
            target node types

        Returns
        -------
        pos_pre, neg_pre : torch.FloatTensor
            edge predictions of positive and negative graph in the decoder
        """

        c_feat, g_feat, c_last, g_last, q = self.encode(enc_graph, adj, ckey, gkey)


        if neg_graph is not None:
            pos_pre, neg_pre = self.decode(pos_graph, neg_graph, c_feat, g_feat, g_last, ckey, gkey)
            return pos_pre, neg_pre, q
        else:
            pos_pre, neg_pre = self.decode(pos_graph, neg_graph, c_feat, g_feat, g_last, ckey, gkey)
            return pos_pre, neg_pre, q






