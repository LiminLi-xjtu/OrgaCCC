import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


flags = tf.app.flags
FLAGS = flags.FLAGS


def normalize(mx):
    rowsum = np.array(mx.sum(1),dtype=np.float32)  
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    # Out-degree normalization of adj (see section 3.3.1 of paper)
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -1).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    return sparse_to_tuple(adj_normalized)

def norm_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    # Out-degree normalization of adj 
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -1).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    return adj_normalized

def construct_feed_dict(adj_normalized_gene, adj_gene, features_gene,
                        adj_normalized_cell, adj_cell, features_cell,
                        placeholders):
    # Construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features_gene']: features_gene})
    feed_dict.update({placeholders['adj_gene']: adj_normalized_gene})
    feed_dict.update({placeholders['adj_orig_gene']: adj_gene})
    feed_dict.update({placeholders['features_cell']: features_cell})
    feed_dict.update({placeholders['adj_cell']: adj_normalized_cell})
    feed_dict.update({placeholders['adj_orig_cell']: adj_cell})
    return feed_dict

def construct_feed_dict_distance(adj_normalized, adj, features, placeholders):
    # Construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict
# Edge Masking for the three directed link prediction tasks

def mask_test_edges_general_link_prediction(adj , test_percent=10., val_percent=5.):
    """
    Task 1: General Directed Link Prediction: get Train/Validation/Test

    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """

    # Remove diagonal elements of adjacency matrix
    adj = adj - sp.dia_matrix((adj.diagonal(), [0]), shape = adj.shape)
    adj.eliminate_zeros()
    edges_positive, _, _ = sparse_to_tuple(adj)

    # Number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # Sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    # positive val edges
    val_edges = edges_positive[val_edge_idx]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    # positive test edges
    test_edges = edges_positive[test_edge_idx]
    # positive train edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis = 0)

    # (Text from philipjackson)
    # The above strategy for sampling without replacement will not work for sampling
    # negative edges on large graphs, because the pool of negative edges
    # is much much larger due to sparsity, therefore we'll use the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT
    # (without replacement is super slow). sample more than we need so we'll probably
    # have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. remove any duplicate elements if there are any
    # 5. remove any diagonal elements
    # 6. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj) # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices
    # Test set
    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')
    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test - len(test_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        coords = np.unique(coords, axis=0)
        np.random.shuffle(coords)
        # step 5:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 6:
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis = 0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    # Validation set
    val_edges_false = np.empty((0,2), dtype = 'int64')
    idx_val_edges_false = np.empty((0,), dtype = 'int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_val - len(val_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        coords = np.unique(coords, axis = 0)
        np.random.shuffle(coords)
        # step 5:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 6:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis=0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # Sanity checks:
    train_edges_linear = train_edges[:,0]*adj.shape[0] + train_edges[:,1]
    test_edges_linear = test_edges[:,0]*adj.shape[0] + test_edges[:,1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0] + val_edges[:,1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0] + val_edges[:,1], test_edges_linear))

    # Re-build train adjacency matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape = adj.shape)

    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false
