from vgae.layers import *
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    """ Model base class """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        # Wrapper for _build()
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass



class SourceTargetGCNModelVAE_gene(Model):
    """
    Source-Target Graph Variational Autoencoder with 2-layer GCN encoder,
    Gaussian distributions and asymmetric inner product decoder
    """
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super().__init__(**kwargs)
        self.inputs = placeholders['features_gene']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj_gene']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder_gene'):
            self.hidden = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = FLAGS.hidden,
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout,
                                             logging = self.logging,
                                             name='e_dense_1')(self.inputs)

            self.z_mean = GraphConvolution(input_dim = FLAGS.hidden,
                                       output_dim = FLAGS.dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging,
                                       name='e_dense_2')(self.hidden)

            self.z_log_std = GraphConvolution(input_dim = FLAGS.hidden,
                                          output_dim = FLAGS.dimension,
                                          adj = self.adj,
                                          act = lambda x: x,
                                          dropout = self.dropout,
                                          logging = self.logging,
                                          name='e_dense_3')(self.hidden)

            self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.dimension]) * tf.exp(self.z_log_std)

            self.reconstructions = SourceTargetInnerProductDecoder(act = lambda x: x,
                                                               logging = self.logging)(self.z)
            self.exp_rec = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu',input_shape=(FLAGS.dimension,)),
                tf.keras.layers.Dense(self.input_dim, activation='relu')
            ])(self.z[:,0:FLAGS.dimension])

class SourceTargetGCNModelVAE_cell(Model):
    """
    Source-Target Graph Variational Autoencoder with 2-layer GCN encoder,
    Gaussian distributions and asymmetric inner product decoder
    """
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super().__init__(**kwargs)
        self.inputs = placeholders['features_cell']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj_cell']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder_cell'):
            self.hidden = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = FLAGS.hidden,
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout,
                                             logging = self.logging,
                                             name='e_dense_1')(self.inputs)

            self.z_mean = GraphConvolution(input_dim = FLAGS.hidden,
                                       output_dim = FLAGS.dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging,
                                       name='e_dense_2')(self.hidden)

            self.z_log_std = GraphConvolution(input_dim = FLAGS.hidden,
                                          output_dim = FLAGS.dimension,
                                          adj = self.adj,
                                          act = lambda x: x,
                                          dropout = self.dropout,
                                          logging = self.logging,
                                          name='e_dense_3')(self.hidden)

            self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.dimension]) * tf.exp(self.z_log_std)

            self.reconstructions = SourceTargetInnerProductDecoder(act = lambda x: x,
                                                               logging = self.logging)(self.z)
            self.exp_rec = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu',input_shape=(FLAGS.dimension,)),
                tf.keras.layers.Dense(self.input_dim, activation='relu')
            ])(self.z[:,0:FLAGS.dimension])

def cross_correlation(x1, x2):
    """
    calculate the cross-view correlation matrix S
    Args:
        Z_v1: the first view embedding
        Z_v2: the second view embedding
    Returns: S
    """
    x1_normalized = tf.nn.l2_normalize(x1, axis=1)  # 对x1进行L2归一化
    x2_normalized = tf.nn.l2_normalize(x2, axis=1)  # 对x2进行L2归一化
    return tf.linalg.matmul(x1_normalized, x2_normalized)


def correlation_reduction_loss(S):
    """
    the correlation reduction loss L: MSE for S and I (identical matrix)
    Args:
        S: the cross-view correlation matrix S
    Returns: L
    """
    diagonal = tf.linalg.diag_part(S)  # 获取矩阵的对角线元素
    off_diagonal = S - tf.linalg.diag(diagonal)  # 获取矩阵的非对角线元素
    return tf.math.reduce_mean(tf.math.pow(diagonal - 1, 2)) + tf.math.reduce_mean(tf.math.pow(off_diagonal, 2)) 

