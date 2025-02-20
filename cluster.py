import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.linalg import eigsh
import scanpy as sc


def spectral_clustering_analysis(adata, adj_reconstructed, k, random_state=42):
    """
    使用谱聚类对细胞进行聚类，并在空间上展示聚类结果。

    参数:
        adata: AnnData 对象，包含细胞和其空间坐标信息
        adj_reconstructed: 邻接矩阵（可以是稀疏矩阵）
        k: 聚类的数量，默认为6
        random_state: 随机种子，确保结果的可重复性，默认为42

    返回:
        返回更新了聚类标签的 adata 对象，并绘制空间图。
    """
    # Step 1: 将邻接矩阵转化为稀疏矩阵
    x_sparse = csr_matrix(adj_reconstructed, dtype=np.float64)

    # Step 2: 计算图的拉普拉斯矩阵（未规范化）
    laplacian = csgraph.laplacian(x_sparse, normed=False)

    # Step 3: 特征值分解，获取前 k 个特征向量
    eigvals, eigvecs = eigsh(laplacian, k=k, which='SM')

    # Step 4: 归一化特征向量
    embeddings = normalize(eigvecs)

    # Step 5: 使用 SpectralClustering 聚类
    spectral_clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=random_state)
    adjacency_matrix = x_sparse.toarray()  # 转化为密集矩阵
    spectral_cluster_labels = spectral_clustering.fit_predict(adjacency_matrix)

    # Step 6: 将聚类结果添加到 adata 对象中
    spectral_cluster = [str(label) for label in spectral_cluster_labels]
    adata.obs["spectral_cluster"] = pd.Categorical(spectral_cluster,
                                                   categories=sorted(set(spectral_cluster), key=lambda x: int(x)))

    # Step 7: 绘制空间图
    palette = sns.color_palette("Set1", k)
    sc.pl.embedding(adata, basis="spatial", color="spectral_cluster", palette=palette, frameon=False, title=None,
                    size=170)

    return adata, embeddings

