import os
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from vgae.preprocessing import *
from vgae.evaluation import *
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

import json
from tools.plot import *
import copy

def freeze_graph(input_checkpoint, output_graph, output_node_names):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
           f.write(output_graph_def.SerializeToString())  # 序列化输出
           model_filename = output_graph

        with gfile.FastGFile(model_filename,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        result = tf.import_graph_def(graph_def, return_elements=[output_node_names+':0'])
        weights = sess.run(result)
        return np.array(weights[0])
    
def get_weight(input_checkpoint):
    e_dense_1_weights = freeze_graph(input_checkpoint, './model/e_dense_1_weights.pb', 'sourcetargetgcnmodelvae_cell/Encoder_cell/e_dense_1_vars/weights')
    e_dense_2_weights = freeze_graph(input_checkpoint, './model/e_dense_2_weights.pb', 'sourcetargetgcnmodelvae_cell/Encoder_cell/e_dense_2_vars/weights')
    e_dense_3_weights = freeze_graph(input_checkpoint, './model/e_dense_3_weights.pb', 'sourcetargetgcnmodelvae_cell/Encoder_cell/e_dense_3_vars/weights')

    return e_dense_1_weights, e_dense_2_weights, e_dense_3_weights


def get_sensitivity(adata, adj_cell, features,test_edges_cell,test_edges_false_cell,input_checkpoint):
    gene_name = adata.var_names
    adj_norm = norm_graph(adj_cell).astype(np.float32)
    e_dense_1_weights, e_dense_2_weights, e_dense_3_weights = get_weight(input_checkpoint)
    e_dense_1_weights = sp.csr_matrix(e_dense_1_weights)
    e_dense_2_weights = sp.csr_matrix(e_dense_2_weights)
    e_dense_3_weights = sp.csr_matrix(e_dense_3_weights)
    sess = tf.Session()

    def single_gene_occlusion(features,test_edges_cell,test_edges_false_cell):
        # Get embeddings with occluded gene expression
        hidden = np.dot(np.dot(adj_norm, features), e_dense_1_weights)
        hidden = (np.abs(hidden) + hidden) / 2.0
        mean = np.dot(np.dot(adj_norm, hidden), e_dense_2_weights)
        std = np.dot(np.dot(adj_norm, hidden), e_dense_3_weights)
        H = mean + tf.random_normal([adj_cell.shape[0], 32]).eval(session=sess) * tf.exp(std.todense()).eval(session=sess)
        # Calculate test score with occluded gene expression
        roc_score, ap_score = compute_scores(test_edges_cell, test_edges_false_cell, np.array(H))
        return roc_score, ap_score

    # Calculate test score with original gene expression 
    roc_score_orig, ap_score_orig = single_gene_occlusion(features,test_edges_cell,test_edges_false_cell)

    # Calculate the test score for each gene in a loop
    single_gene_roc_score = dict()
    single_gene_ap_score = dict()

    for i in range(0,features.shape[1]):
        col_all_roc_score = []
        col_all_ap_score = []
        for j in range(30):
            exp_occlu = copy.deepcopy(features).todense()
            np.random.shuffle(exp_occlu[:,i])
            roc_score, ap_score = single_gene_occlusion(sp.csr_matrix(exp_occlu),test_edges_cell,test_edges_false_cell)
            col_all_roc_score.append(roc_score)
            col_all_ap_score.append(ap_score)
            del exp_occlu
        single_gene_roc_score.update({gene_name[i]: col_all_roc_score})
        single_gene_ap_score.update({gene_name[i]: col_all_ap_score})
        print("complted:" ,i ,gene_name[i])

    # Get gene sensitivity
    occlu_deta_ap = {}
    occlu_deta_roc = {}
    for k,v in single_gene_ap_score.items():
        occlu_deta_ap[k] = np.abs(float(ap_score_orig) - np.array(v).mean())
    for k,v in single_gene_roc_score.items():
        occlu_deta_roc[k] = np.abs(float(roc_score_orig) - np.array(v).mean())

    # Save results
    def write_json(object, filename):
        with open(filename + '.json', 'w') as f:
            json.dump(object, f)

    def output_gene_sensitivity(occlu_deta_score):
        gene_list = sorted(occlu_deta_score.items(), key=lambda item: item[1], reverse=True)
        f = open('Gene_sensitivity.txt', "a")
        for list_mem in gene_list:
            f.write(list_mem[0] + "\t")
            f.write(str(list_mem[1]) + "\n")
        f.close()

    write_json(single_gene_roc_score, 'ob_single_gene_occlusion_roc_score')
    write_json(single_gene_ap_score, 'ob_single_gene_occlusion_ap_score')
    f = open("ob_single_gene_occlusion_score_orig.txt","a")
    f.write(str(roc_score_orig) + " ")
    f.write(str(ap_score_orig) + " ")
    f.close()
    output_gene_sensitivity(occlu_deta_ap)

    plot_top10_gene_sensitivity("top10_gene_sensitivity.pdf",occlu_deta_ap, xlabel='gene name', ylabel='sensitivity', filename='Top10_gene_sensitivity')
    plot_all_gene_sensitivity(occlu_deta_ap, "all_gene_sensitivity.pdf")
