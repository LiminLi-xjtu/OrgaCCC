import numpy as np
import pandas as pd
import anndata

def summarize_cluster(X, clusterid, clusternames, n_permutations=500):
    # Input a sparse matrix of cell signaling and output a pandas dataframe
    # for cluster-cluster signaling
    n = len(clusternames)
    X_cluster = np.empty([n,n], float)
    p_cluster = np.zeros([n,n], float)
    for i in range(n):
        tmp_idx_i = np.where(clusterid==clusternames[i])[0]
        for j in range(n):
            tmp_idx_j = np.where(clusterid==clusternames[j])[0]
            X_cluster[i,j] = X[tmp_idx_i,:][:,tmp_idx_j].mean()
    for i in range(n_permutations):
        clusterid_perm = np.random.permutation(clusterid)
        X_cluster_perm = np.empty([n,n], float)
        for j in range(n):
            tmp_idx_j = np.where(clusterid_perm==clusternames[j])[0]
            for k in range(n):
                tmp_idx_k = np.where(clusterid_perm==clusternames[k])[0]
                X_cluster_perm[j,k] = X[tmp_idx_j,:][:,tmp_idx_k].mean()
        p_cluster[X_cluster_perm >= X_cluster] += 1.0
    p_cluster = p_cluster / n_permutations
    df_cluster = pd.DataFrame(data=X_cluster, index=clusternames, columns=clusternames)
    df_p_value = pd.DataFrame(data=p_cluster, index=clusternames, columns=clusternames)
    return df_cluster, df_p_value

def cluster_communication(
    adata: anndata.AnnData,
    database_name: str = None,
    pathway_name: str = None,
    lr_pair = None,
    clustering: str = None,
    n_permutations: int = 100,
    random_seed: int = 1,
    copy: bool = False
):
    """
    Summarize cell-cell communication to cluster-cluster communication and compute p-values by permutating cell/spot labels.

    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    database_name
        Name of the ligand-receptor database.
        If both pathway_name and lr_pair are None, the cluster signaling through all ligand-receptor pairs is summarized.
    pathway_name
        Name of the signaling pathway.
        If given, the signaling through all ligand-receptor pairs of the given pathway is summarized.
    lr_pair
        A tuple of ligand-receptor pair. 
        If given, only the cluster signaling through this pair is computed.
    clustering
        Name of clustering with the labels stored in ``.obs[clustering]``.
    n_permutations
        Number of label permutations for computing the p-value.
    random_seed
        The numpy random_seed for reproducible random permutations.
    copy
        Whether to return a copy of the :class:`anndata.AnnData`.
    
    Returns
    -------
    adata : anndata.AnnData
        Add cluster-cluster communication matrix to 
        ``.uns['commot_cluster-databaseX-clustering-ligA-recA']``
        for the ligand-receptor database named 'databaseX' and the cell clustering 
        named 'clustering' through the ligand-receptor pair 'ligA' and 'recA'.
        The first object is the communication score matrix and the second object contains
        the corresponding p-values.
        If copy=True, return the AnnData object and return None otherwise.

    """
    np.random.seed(random_seed)

    assert database_name is not None, "Please at least specify database_name."

    celltypes = list( adata.obs[clustering].unique() )
    celltypes.sort()
    for i in range(len(celltypes)):
        celltypes[i] = str(celltypes[i])
    clusterid = np.array(adata.obs[clustering], str)
    obsp_names = []
    if not lr_pair is None:
        obsp_names.append(database_name+'-'+lr_pair[0]+'-'+lr_pair[1])
    elif not pathway_name is None:
        obsp_names.append(database_name+'-'+pathway_name)
    else:
        obsp_names.append(database_name+'-total-total')
    # name_mat = adata.uns['commot-'+pathway_name+'-info']['df_ligrec'].values
    # name_mat = np.concatenate((name_mat, np.array([['total','total']],str)), axis=0)
    for i in range(len(obsp_names)):
        S = adata.obsp['orgaccc-'+obsp_names[i]]
        tmp_df, tmp_p_value = summarize_cluster(S,
            clusterid, celltypes, n_permutations=n_permutations)
        adata.uns['orgaccc_cluster-'+clustering+'-'+obsp_names[i]] = {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value}
    
    return adata if copy else None


def get_cluster_communication_network(
    adata: anndata.AnnData,
    uns_names: list = None,
    clustering: str = None,
    quantile_cutoff: float = 0.99,
    p_value_cutoff: float = 0.05,
    self_communication_off: bool = False,
):
    """
    Plot cluster-cluster communication as network.

    .. image:: cluster_communication.png
        :width: 500pt


    Parameters
    ----------
    adata
        The data matrix of shape ``n_obs`` × ``n_var``.
        Rows correspond to cells or positions and columns to genes.
    uns_names
        A list of keys for the cluster-level CCC stored in ``.uns`` data slot.
        When more than one is given, the average score will be used.
        For example, ``'commot_cluster-leiden-cellchat-Fgf7-Fgfr1'`` will plot the CCC inferred 
        with 'cellchat' LR database through 'Fgf7-Fgfr1' LR pair and for clustering stored in ``.obs['leiden']``.
    clustering
        Name of the clustering. The cluster labels should be available in ``.obs[clustering]``.
    quantile_cutoff
        The quantile cutoff for including an edge. Set to 1 to disable this criterion.
        The quantile_cutoff and p_value_cutoff works in the "or" logic to avoid missing
        significant signaling connections.
    p_value_cutoff
        The cutoff of p-value to include an edge.
    self_communication_off
        Whether to exclude self communications in the visualization.
    filename
        Filename for saving the figure. Set the name to end with '.pdf' or 'png'
        to specify format.
    nx_node_size
        Size of node representing clusters.
    nx_node_cmap
        The discrete color map to use for clusters. Choices: 
        'Plotly', 'Alphabet', 'Light24', 'Dark24'. Recommend to use 'Plotly'
        for ten clusters or fewer and 'Alphabet' for 10-24 clusters.
    nx_pos_idx
        Coordinates to use for the 2D plot.
    nx_node_pos
        'cluster', the predicted spatial location of clusters will be used. 
        If setting to 'cluster', run the function :func:`cluster_position` first to set the cluster positions.
        If None, the 'dot' layout from Graphviz package will be used.
    nx_edge_width_lb_quantile
        The quantile of communication connections to set for the lower bound of edge
        width.
    nx_edge_width_ub_quantile
        The quantile of communication connections to set for the upper bound of edge
        width.
    nx_edge_width_min
        Minimum width for plotted edges.
    nx_edge_width_max
        Maximum width for plotted edges.
    nx_edge_color
        If 'node', the color of an edge will be the same as the source node.
        If an array of numbers between [0,1], the nx_edge_colormap will be used
        to determine the edge colors.
    nx_edge_colormap
        The color map to use when nx_edge_color is an array of weights.
    nx_bg_pos
        Whether to plot the cells/positions as spatial background.
        Set to False when not using the spatial layout of clusters.
    nx_bg_color
        Color of the spatial background.
    nx_bg_ndsize
        Node size of the spatial background.

    """
    
    X_tmp = adata.uns[uns_names[0]]['communication_matrix'].copy()
    labels = list( X_tmp.columns.values )
    X = np.zeros_like(X_tmp.values, float)
    for i in range(len(uns_names)):
        X_tmp = adata.uns[uns_names[i]]['communication_matrix'].values.copy()
        p_values_tmp = adata.uns[uns_names[i]]['communication_pvalue'].values.copy()
        if not quantile_cutoff is None:
            cutoff = np.quantile(X_tmp.reshape(-1), quantile_cutoff)
        else:
            cutoff = np.inf
        tmp_mask = ( X_tmp < cutoff ) * ( p_values_tmp > p_value_cutoff )
        X_tmp[tmp_mask] = 0
        X = X + X_tmp
    X = X / len(uns_names)
    if self_communication_off:
        for i in range(X.shape[0]):
            X[i,i] = 0
            
    return X