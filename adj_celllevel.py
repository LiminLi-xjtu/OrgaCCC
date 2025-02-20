from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics  import accuracy_score
from sklearn import metrics
# from munkres import Munkres, print_matrix
import numpy as np
import copy
from scipy.special import expit


class select_optimal_threshold():
    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def select(self, emb):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        dim = 32
        adj_rec = np.zeros((len(emb),len(emb)))
        for i in range(len(emb)):
            for j in range(len(emb)):
                adj_rec[i,j] = sigmoid(emb[i,0:int(dim/2)].dot(emb[j,int(dim/2):dim].T))

        for i in range(0, adj_rec.shape[0]):
            adj_rec[i,i] = 0
        preds = []
        pos = []
        
        for e in self.edges_pos:
            preds.append(sigmoid(emb[e[0],0:int(dim/2)].dot(emb[e[1],int(dim/2):dim].T)))

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(emb[e[0],0:int(dim/2)].dot(emb[e[1],int(dim/2):dim].T)))

        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

        all_acc_score = {}
        max_acc_score = 0
        optimal_threshold = 0
        for threshold in np.arange(0.01,1,0.005):
            preds_tem = copy.deepcopy(preds)
            preds_tem = [np.array(matrix).tolist() for matrix in preds_tem]
            preds_tem = [item for sublist in preds_tem for item in sublist]
            preds_tem = [item for sublist in preds_tem for item in sublist]
            preds_neg_tem = copy.deepcopy(preds_neg)
            preds_neg_tem = [np.array(matrix).tolist() for matrix in preds_neg_tem]
            preds_neg_tem = [item for sublist in preds_neg_tem for item in sublist]
            preds_neg_tem = [item for sublist in preds_neg_tem for item in sublist]

            preds_all = np.hstack([preds_tem, preds_neg_tem])
            preds_all = (preds_all>threshold).astype('int')
            acc_score = accuracy_score(labels_all, preds_all)
            all_acc_score[threshold] = acc_score
            if acc_score > max_acc_score:
                max_acc_score = acc_score
                optimal_threshold = threshold

        for i in range(0, adj_rec.shape[0]):
            adj_rec[i,i] = 0

        adj_rec_1 = copy.deepcopy(adj_rec)
        adj_rec_1 = (adj_rec_1>optimal_threshold).astype('int')
        for j in range(0, adj_rec_1.shape[0]):
            adj_rec_1[j,j] = 0



        return adj_rec, adj_rec_1, all_acc_score, max_acc_score, optimal_threshold




