from __future__ import division
from __future__ import print_function
from tools.adj import *
from vgae.model import *
from vgae.optimizer import Orga_Optimizer_all
from tools.celltype_communication import *
from tools.adj_celllevel import *
from tools.cluster import *
from tools.sensitivity import *
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import time
import umap
import matplotlib.pyplot as plt
import os
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn import preprocessing
seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS


# Model parameters
flags.DEFINE_float('dropout', 0.0001, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs in training.')
# flags.DEFINE_boolean('features', False, 'Include node features or not in GCN')
flags.DEFINE_float('lamb', 1., 'lambda parameter from Gravity AE/VAE models \
                                as introduced in section 3.5 of paper, to \
                                balance mass and proximity terms')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate (with Adam)')
flags.DEFINE_integer('hidden', 64, 'Number of units in GCN hidden layer.')
flags.DEFINE_integer('dimension', 32, 'Dimension of GCN output: \
- equal to embedding dimension for standard AE/VAE and source-target AE/VAE \
- equal to (embedding dimension - 1) for gravity-inspired AE/VAE, as the \
last dimension captures the "mass" parameter tilde{m}')
flags.DEFINE_boolean('normalize', False, 'Whether to normalize embedding \
                                          vectors of gravity models')
flags.DEFINE_float('epsilon', 0.001, 'Add epsilon to distances computations \
                                       in gravity models, for numerical \
                                       stability')
# Experimental setup parameters
flags.DEFINE_integer('nb_run', 5, 'Number of model run + test')
flags.DEFINE_float('prop_val', 5., 'Proportion of edges in validation set \
                                   (for Task 1)')
flags.DEFINE_float('prop_test', 10., 'Proportion of edges in test set \
                                      (for Tasks 1 and 2)')
flags.DEFINE_boolean('validation', False, 'Whether to report validation \
                                           results  at each epoch (for \
                                           Task 1)')
flags.DEFINE_boolean('verbose', True, 'Whether to print comments details.')

# Load graph dataset
adata = sc.read_h5ad(r"data/cortex.h5ad")
adata.var_names_make_unique

################准备基因数据##################
gene_adj, _ = load_geneadj(adata)
gene_adj = gene_adj.values
adj_init_gene = sp.csr_matrix(gene_adj)
exp = adata.X
exp = pd.DataFrame(exp)
features_gene = adata.X.T
scaler_gene = preprocessing.StandardScaler().fit(features_gene)
features_gene = scaler_gene.transform(features_gene)
features_gene = sp.csr_matrix(features_gene)

###############准备细胞数据###################
cell_adj = load_celladj(adata, distance=200)
np.count_nonzero(cell_adj)
adj_init_cell = sp.csr_matrix(cell_adj)
features_cell = adata.X
scaler_cell = preprocessing.StandardScaler().fit(features_cell)
features_cell = scaler_cell.transform(features_cell)
features_cell = sp.csr_matrix(features_cell)

# The entire training process is repeated FLAGS.nb_run times

os.mkdir("model")

for i in range(FLAGS.nb_run):
    print("Masking test edges...")
    ####基因####
    adj_gene, val_edges_gene, val_edges_false_gene, test_edges_gene, test_edges_false_gene = \
        mask_test_edges_general_link_prediction(adj_init_gene, FLAGS.prop_test,
                                                FLAGS.prop_val)
    ####细胞####
    adj_cell, val_edges_cell, val_edges_false_cell, test_edges_cell, test_edges_false_cell = \
        mask_test_edges_general_link_prediction(adj_init_cell, FLAGS.prop_test,
                                                FLAGS.prop_val)
    # Preprocessing and initialization
    print("Preprocessing and Initializing...")
    # Compute number of nodes
    num_nodes_gene = adj_gene.shape[0]
    num_nodes_cell = adj_cell.shape[0]

    ####基因####
    features_gene = sparse_to_tuple(features_gene)
    num_features_gene = features_gene[2][1]
    features_nonzero_gene = features_gene[1].shape[0]

    ####细胞####
    features_cell = sparse_to_tuple(features_cell)
    num_features_cell = features_cell[2][1]
    features_nonzero_cell = features_cell[1].shape[0]

    # Define placeholders
    placeholders = {
        'features_gene': tf.sparse_placeholder(tf.float32),
        'adj_gene': tf.sparse_placeholder(tf.float32),
        'adj_orig_gene': tf.sparse_placeholder(tf.float32),

        'features_cell': tf.sparse_placeholder(tf.float32),
        'adj_cell': tf.sparse_placeholder(tf.float32),
        'adj_orig_cell': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # Create model

    model_gene = SourceTargetGCNModelVAE_gene(placeholders, num_features_gene, num_nodes_gene,
                                              features_nonzero_gene)

    model_cell = SourceTargetGCNModelVAE_cell(placeholders, num_features_cell, num_nodes_cell,
                                              features_nonzero_cell)

    # Optimizer (see tkipf/gae original GAE repository for details)
    pos_weight_gene = float(adj_gene.shape[0] * adj_gene.shape[0] - adj_gene.sum()) / adj_gene.sum()
    norm_gene = adj_gene.shape[0] * adj_gene.shape[0] / float((adj_gene.shape[0] * adj_gene.shape[0]
                                                               - adj_gene.sum()) * 2)

    pos_weight_cell = float(adj_cell.shape[0] * adj_cell.shape[0] - adj_cell.sum()) / adj_cell.sum()
    norm_cell = adj_cell.shape[0] * adj_cell.shape[0] / float((adj_cell.shape[0] * adj_cell.shape[0]
                                                               - adj_cell.sum()) * 2)
    # Normalization and preprocessing on adjacency matrix
    adj_norm_gene = preprocess_graph(adj_gene)
    adj_norm_cell = preprocess_graph(adj_cell)

    adj_label_gene = sparse_to_tuple(adj_gene + sp.eye(adj_gene.shape[0]))
    adj_label_cell = sparse_to_tuple(adj_cell + sp.eye(adj_cell.shape[0]))

    # Optimizer for Variational Autoencoders
    opt = Orga_Optimizer_all(preds_gene=model_gene.reconstructions,
                             labels_gene=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig_gene'],
                                                                              validate_indices=False), [-1]),
                             model_gene=model_gene,
                             num_nodes_gene=num_nodes_gene,
                             pos_weight_gene=pos_weight_gene,
                             norm_gene=norm_gene,
                             preds_cell=model_cell.reconstructions,
                             labels_cell=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig_cell'],
                                                                              validate_indices=False), [-1]),
                             model_cell=model_cell,
                             num_nodes_cell=num_nodes_cell,
                             pos_weight_cell=pos_weight_cell,
                             norm_cell=norm_cell,
                             exp=exp)
    # Saver
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Model training
    print("Training...")
    # Flag to compute total running time
    t_start = time.time()
    for epoch in range(FLAGS.epochs):
        # Flag to compute running time for each epoch
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm_gene, adj_label_gene, features_gene,
                                        adj_norm_cell, adj_label_cell, features_cell,
                                        placeholders)

        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy_gene, opt.accuracy_cell],
                        feed_dict=feed_dict)
        # Compute average loss
        avg_cost = outs[1]

        # Display epoch information
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "time=", "{:.5f}".format(time.time() - t))

        saver.save(sess, './model/', global_step=epoch + 1)
        # Validation (implemented for Task 1 only)
        if FLAGS.validation:
            feed_dict.update({placeholders['dropout']: 0})
            emb_gene = sess.run(model_gene.z_mean, feed_dict=feed_dict)
            emb_cell = sess.run(model_cell.z_mean, feed_dict=feed_dict)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            val_roc, val_ap = compute_scores(val_edges_cell, val_edges_false_cell, emb_cell)
            print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))

    # Get embedding from model
    emb_gene = sess.run(model_gene.z_mean, feed_dict=feed_dict)
    emb_cell = sess.run(model_cell.z_mean, feed_dict=feed_dict)

    # Test model
    print("Testing model...")
    # Compute ROC and AP scores on test sets
    roc_score, ap_score = compute_scores(test_edges_cell, test_edges_false_cell, emb_cell)
    print("AUC scores\n", roc_score)
    print("AP scores \n", ap_score)


################################重构细胞通讯图################################
emb = sess.run(model_cell.z_mean, feed_dict=feed_dict)
dim = FLAGS.dimension  # Embedding dimension
epsilon = FLAGS.epsilon
adj_rec = np.zeros((len(emb), len(emb)))
for i in range(len(emb)):
    for j in range(len(emb)):
        adj_rec[i, j] = sigmoid(emb[i, 0:int(dim / 2)].dot(emb[j, int(dim / 2):dim].T))
for i in range(0, adj_rec.shape[0]):
    adj_rec[i, i] = 0

adj_reconstructed_prob, adj_reconstructed, all_acc_score, max_acc_score, optimal_threshold = select_optimal_threshold(test_edges_cell, test_edges_false_cell).select(np.matrix(emb))


################################细胞类型对通讯结果X################################
adata.obsp['orgaccc-cellchat-total-total'] =  pd.DataFrame(adj_rec)
cluster_communication(adata, database_name='cellchat',clustering='celltype',n_permutations=100)
X = get_cluster_communication_network(adata, uns_names=['orgaccc_cluster-celltype-cellchat-total-total'],clustering='celltype',  p_value_cutoff = 5e-2)


################################聚类分析################################
k=6
adata, embeddings = spectral_clustering_analysis(adata, adj_reconstructed, k)

#### 使用 UMAP 降维并绘图
latent_feature_umap = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(embeddings)
# 确保 spectral_cluster 是一致的分类顺序
spectral_cluster = pd.Categorical(adata.obs["spectral_cluster"], categories=sorted(set(adata.obs["spectral_cluster"]), key=lambda x: int(x)))
# 绘制 UMAP 图，按聚类结果上色
plt.figure(figsize=(8, 6))
sns.scatterplot(x=latent_feature_umap[:, 0], y=latent_feature_umap[:, 1], hue=spectral_cluster, legend="full", palette=sns.color_palette("Set1", k))
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Cluster")
plt.show()

####PAGA分析
adata.obsm["X"] = embeddings
sc.pp.neighbors(adata, use_rep='X')
sc.tl.paga(adata, groups='spectral_cluster')
sc.pl.paga(adata, threshold=0.03, show=True, node_size_scale=5, edge_width_scale=3, fontsize=20)


################################基因敏感性分析################################
#需要较长时间
#input_checkpoint = 'model/-1000' #根据训练结果修改
#get_sensitivity(adata, adj_cell, features_cell,test_edges_cell,test_edges_false_cell,input_checkpoint)
