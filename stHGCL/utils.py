
import dgl
import numpy as np
import pandas as pd
import random
import os
import ot
import torch
import warnings
warnings.filterwarnings('ignore')
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def setup_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def kmeans(adata, n_clusters=7, use_rep=None):
    k_means = KMeans(n_clusters, n_init=20, random_state=0)
    y_pred = k_means.fit_predict(adata.obsm[use_rep])
    adata.obs['kmeans'] = y_pred
    adata.obs['domain'] = adata.obs['kmeans'].astype(str).astype('category')
    return adata

def calculate_metric(pred, label):
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 4)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 4)

    return nmi, ari


def mclust_R(adata, n_clusters, modelNames='EEE', used_obsm='feat', random_seed=666):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata
