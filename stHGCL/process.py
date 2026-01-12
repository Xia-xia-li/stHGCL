import numpy as np
import scanpy as sc
import pandas as pd


def adata_preprocess(adata_vis, min_cells=50, min_counts=10, pca_n_comps=200, hight_var=2000):
    adata_vis.layers['count'] = adata_vis.X
    sc.pp.filter_genes(adata_vis, min_cells=min_cells)
    sc.pp.filter_genes(adata_vis, min_counts=min_counts)
    sc.pp.normalize_total(adata_vis, target_sum=1e6)
    # adata_vs = spatially_variable_genes(adata_vis, mode=choose, n_top_genes=3000, subset=True)
    # sc.pp.log1p(adata_vis)
    if hight_var == None:
        X_ = adata_vis.X
        adata_vis.obsm['X_'] = X_
    else:
        sc.pp.highly_variable_genes(adata_vis, flavor="seurat_v3", layer='count', n_top_genes=hight_var)
        adata_vis = adata_vis[:, adata_vis.var['highly_variable'] == True]
        X_ = adata_vis.X
        adata_vis.obsm['X_'] = X_

    return adata_vis


def preprocess(adata, min_cells=50, min_counts=10, pca_n_comps=200, hight_var=2000,Resolution = 'spot'):

    adata = process_noground(adata, min_cells, min_counts, pca_n_comps, hight_var,Resolution)
    return adata


def process_noground(adata, min_cells=50, min_counts=10, pca_n_comps=200, hight_var=2000 , Resolution = 'spot'):
    # This data could be downloaded frogle.com/drive/fo
    # add ground_truth
    if Resolution == 'spot':
         #adata.X = adata.X.toarray().astype('float')
        adata.X = adata.X.toarray().astype('float')
        # adata.X = adata.X.astype('float')
        # adata.X = adata.X.astype('float')
    else:
        adata.X = adata.X.astype('float')
    adata = adata_preprocess(adata, min_cells, min_counts, pca_n_comps, hight_var)
    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs['sz_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata)
    adata.var['gs_factor'] = np.max(adata.X, axis=0, keepdims=True).reshape(-1)
    p_counts = adata.X
    adata.obsm['p_counts'] = p_counts

    print(adata)
    return adata




def gene_process(file_fold, adata, GT='layer_guess'):
    GT = 'layer_guess'
    raw = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    raw.var_names_make_unique()
    # This data could be downloaded frogle.com/drive/fo
    # add ground_truth
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta[GT]
    raw.layers['count'] = raw.X.toarray()
    raw.X = raw.X.toarray().astype('float')
    raw.obs['ground_truth'] = df_meta_layer.values
    raw = raw[~pd.isnull(raw.obs['ground_truth'])]
    raw.obs['x_pixel'] = raw.obsm["spatial"][:, 0].tolist()
    raw.obs['y_pixel'] = raw.obsm["spatial"][:, 1].tolist()
    raw.obs['x_array'] = raw.obs['array_row']
    raw.obs['y_array'] = raw.obs['array_col']
    raw.obs['pred'] = adata.obs['kmeans']
    x_array = raw.obs["x_array"].tolist()
    y_array = raw.obs["y_array"].tolist()
    x_pixel = raw.obs["x_pixel"].tolist()
    y_pixel = raw.obs["y_pixel"].tolist()
    min_cells = 50
    min_counts = 10
    spg.prefilter_genes(raw, min_cells=3)  # avoiding all genes are zeros
    spg.prefilter_specialgenes(raw)
    sc.pp.normalize_per_cell(raw)
    sc.pp.log1p(raw)
    sc.pp.highly_variable_genes(raw, flavor="seurat_v3", layer='count', n_top_genes=2000)
    raw = raw[:, raw.var['highly_variable'] == True]
    raw.obs['cs_factor'] = adata.obs['cs_factor'].values
    raw.var['gs_factor'] = adata.var['gs_factor'].values
    raw.obsm['feat'] = adata.obsm['feat']
    raw.varm['feat'] = adata.varm['feat']
    return raw, x_array, y_array, x_pixel, y_pixel