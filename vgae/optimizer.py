
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from vgae.model import correlation_reduction_loss, cross_correlation


flags = tf.app.flags
FLAGS = flags.FLAGS

        
class Orga_Optimizer_all(object):
    """ Optimizer for VGAE+VAE_X"""
    def __init__(self, preds_gene, labels_gene, model_gene, num_nodes_gene, pos_weight_gene, norm_gene,
                 preds_cell, labels_cell, model_cell, num_nodes_cell, pos_weight_cell, norm_cell,
                 exp):
        #, feed_dictp
        preds_sub_gene =  preds_gene
        preds_sub_cell = preds_cell
        labels_sub_gene = labels_gene
        labels_sub_cell = labels_cell
        exp = tf.constant(exp.values, dtype=tf.float32)
        self.cross_gene = tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub_gene ,
                                                     targets = labels_sub_gene,
                                                     pos_weight = pos_weight_gene)
        self.cross_gene = tf.where(tf.math.is_nan(self.cross_gene), tf.zeros_like(self.cross_gene), self.cross_gene)
        self.cost_gene = norm_gene * tf.reduce_mean(self.cross_gene)
        self.cost_cell = norm_cell * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub_cell,
                                                     targets = labels_sub_cell,
                                                     pos_weight = pos_weight_cell))
        # Adam Optimizer
        #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=FLAGS.learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        # Latent loss
        self.log_lik_gene = self.cost_gene
        self.log_lik_cell = self.cost_cell
        
        self.kl_gene = (0.5 / num_nodes_gene) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * model_gene.z_log_std \
                                               - tf.square(model_gene.z_mean) \
                                               - tf.square(tf.exp(model_gene.z_log_std)), 1))
        self.kl_cell = (0.5 / num_nodes_cell) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * model_cell.z_log_std \
                                               - tf.square(model_cell.z_mean) \
                                               - tf.square(tf.exp(model_cell.z_log_std)), 1))
       
        
        self.cost_gene -= self.kl_gene
        self.cost_cell -= self.kl_cell
        
        self.x_g = tf.reduce_mean(tf.square(model_gene.exp_rec - tf.transpose(exp)))
        self.x_c = tf.reduce_mean(tf.square(model_cell.exp_rec - exp))
        
        self.l_x = correlation_reduction_loss(cross_correlation(model_gene.exp_rec, model_cell.exp_rec))
        
        
        self.cost = self.cost_gene + self.cost_cell + self.x_g + self.x_c + self.l_x
        
        
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        
        self.correct_prediction_gene = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub_gene), 0.5), tf.int32),
                                              tf.cast(labels_sub_gene, tf.int32))      
        self.correct_prediction_cell = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub_cell), 0.5), tf.int32),
                                              tf.cast(labels_sub_cell, tf.int32))
            
        self.accuracy_gene = tf.reduce_mean(tf.cast(self.correct_prediction_gene, tf.float32))
        self.accuracy_cell = tf.reduce_mean(tf.cast(self.correct_prediction_cell, tf.float32))
  