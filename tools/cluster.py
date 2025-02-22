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
    Perform spectral clustering on cells and visualize the clustering results in spatial coordinates.

    Parameters:
        adata: AnnData object containing cells and their spatial coordinates.
        adj_reconstructed: Adjacency matrix (can be a sparse matrix).
        k: Number of clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Returns the updated adata object with clustering labels and plots the spatial graph.
    """
    # Step 1: Convert the adjacency matrix to a sparse matrix
    x_sparse = csr_matrix(adj_reconstructed, dtype=np.float64)

    # Step 2: Compute the graph Laplacian (unnormalized)
    laplacian = csgraph.laplacian(x_sparse, normed=False)

    # Step 3: Perform eigenvalue decomposition to get the top k eigenvectors
    eigvals, eigvecs = eigsh(laplacian, k=k, which='SM')

    # Step 4: Normalize the eigenvectors
    embeddings = normalize(eigvecs)

    # Step 5: Perform spectral clustering
    spectral_clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=random_state)
    adjacency_matrix = x_sparse.toarray()  # Convert to dense matrix
    spectral_cluster_labels = spectral_clustering.fit_predict(adjacency_matrix)

    # Step 6: Add clustering results to the adata object
    spectral_cluster = [str(label) for label in spectral_cluster_labels]
    adata.obs["spectral_cluster"] = pd.Categorical(spectral_cluster,
                                                   categories=sorted(set(spectral_cluster), key=lambda x: int(x)))
    

    return adata, embeddings
