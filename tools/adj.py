import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def load_geneadj(adata):
    adata.var_names_make_unique()
    data_genes = adata.var_names

    df_cellchat = pd.read_csv(r'../LRdatabase/CellChat/CellChatDB.ligrec.mouse.csv', index_col=0)
    
    df_cellchat = df_cellchat[df_cellchat.iloc[:, 3] == 'Secreted Signaling']

    # 是多聚体
    tmp_ligs = list(set(df_cellchat.iloc[:, 0]))
    tmp_recs = list(set(df_cellchat.iloc[:, 1]))
    avail_ligs = []
    avail_recs = []
    for tmp_lig in tmp_ligs:
        lig_genes = set(tmp_lig.split('_'))
        if lig_genes.issubset(data_genes):
            avail_ligs.append(tmp_lig)
    for tmp_rec in tmp_recs:
        rec_genes = set(tmp_rec.split('_'))
        if rec_genes.issubset(data_genes):
            avail_recs.append(tmp_rec)
    ligs = avail_ligs
    recs = avail_recs
    
    gene_adj = pd.DataFrame(np.zeros((adata.shape[1], adata.shape[1])), columns=data_genes, index=data_genes)

    for i in range(len(df_cellchat)):
        tmp_lig = df_cellchat.iloc[i, 0]
        tmp_rec = df_cellchat.iloc[i, 1]
        if tmp_lig in ligs and tmp_rec in recs:
            for a in range(len(tmp_lig.split('_'))):
                for b in range(len(tmp_rec.split('_'))):
                    gene_adj.loc[tmp_lig.split('_')[a], tmp_rec.split('_')[b]] = 1.0
    
    return gene_adj, np.count_nonzero(gene_adj)
    
def load_celladj(adata, distance):
    dis_mat = distance_matrix(adata.obsm["spatial"], adata.obsm["spatial"])
    #距离200以内的细胞建立边
    ######距离还可以试着改改
    dis_mat[dis_mat <= distance] = 1
    dis_mat[dis_mat > distance] = 0
    np.fill_diagonal(dis_mat, 0)
    cell_adj = dis_mat
    cell_adj = np.nan_to_num(cell_adj)
    return cell_adj


