from time import time
import numpy as np
import platform
from sklearn.metrics import log_loss
from sklearn.utils.sparsefuncs import mean_variance_axis
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, Flatten, Reshape, Conv3D, Conv3DTranspose, MaxPooling2D, Dropout, GlobalMaxPooling2D
from tensorflow.keras.layers import Layer, InputSpec, Input, Dense, Multiply, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling, Zeros, Constant, GlorotNormal, GlorotUniform, \
    LecunUniform, LecunNormal, Orthogonal, RandomNormal, RandomUniform, TruncatedNormal, HeNormal, HeUniform, Identity, Initializer
from tensorflow.keras.regularizers import Regularizer, l1, l2, l1_l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA, SparsePCA
from math import log
import Nmetrics
import matplotlib.pyplot as plt


def FAE(dims, act='relu', view=1, data='data'):
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    input_name = 'v'+str(view)+'_'
    # input
    x = Input(shape=(dims[0],), name='input' + str(view))
    h = x
    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name=input_name+'encoder_%d' % i)(h)
 
    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='embedding' + str(view))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name=input_name+'decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name=input_name+'decoder_0')(y)

    return Model(inputs=x, outputs=y, name=input_name+'Fae'), Model(inputs=x, outputs=h, name=input_name+'Fencoder')


def MAE(view=2, view_shape=[], dim=10, data='data'):
    ae = []
    encoder = []
    for v in range(view):
        ae_tmp, encoder_tmp = FAE(dims=[view_shape[v][0], 500, 500, 2000, dim], view=v + 1, data=data)
        ae.append(ae_tmp)
        encoder.append(encoder_tmp)

    return ae, encoder


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2    
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MvDEC(object):
    def __init__(self,
                 n_clusters=10,
                 alpha=1.0, view_shape=[], dim=10, data='data'):

        super(MvDEC, self).__init__()

        self.view_shape = view_shape
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = False
        # prepare model
        self.view = len(view_shape)
        # print(len(view_shape))

        self.AEs, self.encoders = MAE(view=self.view, view_shape=self.view_shape, dim=dim, data=data)
    
        Input = []
        Output = []
        Input_e = []
        Output_e = []
        clustering_layer = []

        for v in range(self.view):
                Input.append(self.AEs[v].input)
                Output.append(self.AEs[v].output)
                Input_e.append(self.encoders[v].input)
                Output_e.append(self.encoders[v].output)
                clustering_layer.append(ClusteringLayer(self.n_clusters, name='clustering'+str(v+1))(self.encoders[v].output))

        self.autoencoder = Model(inputs=Input, outputs=Output)    # xin _ xout

        self.encoder = Model(inputs=Input_e, outputs=Output_e)   # xin _ q

        Output_m = []
        for v in range(self.view):
            Output_m.append(clustering_layer[v])
            Output_m.append(Output[v])
        self.model = Model(inputs=Input, outputs=Output_m)   # xin _ q _ xout

    def pretrain(self, x, y, optimizer='adam', epochs=200, batch_size=256,
                 save_dir='results/temp', verbose=0):
        print('Begin pretraining: ', '-' * 60)
        multi_loss = []
        for view in range(len(x)):
            multi_loss.append('mse')
        self.autoencoder.compile(optimizer=optimizer, loss=multi_loss)
        save = '/ae_weights.h5'
        # begin pretraining
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=verbose)
        self.autoencoder.save_weights(save_dir + save)
        print('Pretrained weights are saved to ' + save_dir + save)
        # self.pretrained = True
        print('End pretraining: ', '-' * 60)

    def load_weights(self, weights):  # load weights of models
        self.model.load_weights(weights)

    def predict_label(self, x):  # predict cluster labels using the output of clustering layer
        input_dic = {}
        for view in range(len(x)):
            input_dic.update({'input' + str(view+1): x[view]})
        Q_and_X = self.model.predict(input_dic, verbose=0)
        y_pred = []
        for view in range(len(x)):
            # print(view)
            y_pred.append(Q_and_X[view*2].argmax(1))
        
        y_q = Q_and_X[(len(x)-1)*2]
        for view in range(len(x) - 1):
            y_q += Q_and_X[view*2]

        # y_q = y_q/len(x)
        y_mean_pred = y_q.argmax(1)
        return y_pred, y_mean_pred

    @staticmethod    
    def target_distribution(q):
        t = 2
        weight = q ** t
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss=['kld', 'mse'], loss_weights=[0.1, 1.0]):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def train_on_batch(self, xin, yout, sample_weight=None):
        return self.model.train_on_batch(xin, yout, sample_weight)

    def new_fit(self, arg, x, y, maxiter=2e4, batch_size=256, tol=1e-3,
            UpdateCoo=200, save_dir='./results/tmp'):
        print('Begin clustering:', '-' * 60)
        print('Update Coo:', UpdateCoo)
        save_interval = int(maxiter)  # only save the initial and final model
        print('Save interval', save_interval)
        # Step 1: initialize cluster centers using k-means
        t1 = time()
        ting = time() - t1
        # print(ting)

        time_record = []
        time_record.append(int(ting))
        # print(time_record)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)

        input_dic = {}
        for view in range(len(x)):
            input_dic.update({'input' + str(view + 1): x[view]})
        features = self.encoder.predict(input_dic)

        y_pred = []
        center = []

        from numpy import hstack
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        # --------------------------------------------
        for view in range(len(x)):
            y_pred.append(kmeans.fit_predict(features[view]))
            center.append([kmeans.cluster_centers_])
        # --------------------------------------------

        for view in range(len(x)):
            print('Start-' + str(view + 1) + ':')
            from Nmetrics import test
            test(y[view], y_pred[view])
        y_pred_last = []
        y_pred_sp = []
        for view in range(len(x)):
            y_pred_last.append(y_pred[view])
            y_pred_sp.append(y_pred[view])

        print('Initializing cluster centers with K-means.')
        for view in range(len(x)):
            self.model.get_layer(name='clustering' + str(view + 1)).set_weights(center[view])

        # Step 2: deep clustering

        index_array = np.arange(x[0].shape[0])
        index = 0

        Loss = []
        avg_loss = []
        kl_loss = []
        for view in range(len(x)):
            Loss.append(0)
            avg_loss.append(0)
            kl_loss.append(100000)

        update_interval = arg.UpdateCoo

        ACC = []
        NMI = []
        ARI = []
        vACC = []
        vNMI = []
        vARI = []
        MVKLLoss = []
        ite = 0

        initial_flag = 0

        while True:
            if ite % update_interval == 0:
                print('\n')
                for view in range(len(x)):
                    avg_loss[view] = Loss[view] / update_interval
                    kl_loss[view] = kl_loss[view] / update_interval

                Q_and_X = self.model.predict(input_dic)

                for view in range(len(x)):
                    # print(Q_and_X[view * 2][0])
                    y_pred_sp[view] = Q_and_X[view * 2].argmax(1)

                features = self.encoder.predict(input_dic)

                uuu = []
                for view in range(len(x)):
                    muu = self.model.get_layer(name='clustering' + str(view + 1)).get_weights()
                    # print(muu)
                    uuu.append(muu)
                # np.save(save_dir + '/Features/' + str(ite) + '.npy', features)
                # np.save(save_dir + '/Mu/' + str(ite) + '.npy', uuu)

                n_features = []
                weights = []
                sum = 0
                for view in range(len(x)):
                    MU = min_max_scaler.fit_transform(
                        self.model.get_layer(name='clustering' + str(view + 1)).get_weights()[0])
                    # print(MU.shape)
                    # print(MU.var())
                    sum += MU.var()
                    weights.append(MU.var())

                weights = 1 + np.log2(1 + weights / sum)

                # print(weights)

                for view in range(len(x)):
                    # n_features.append(features[view])
                    if arg.dataset == "Caltech":
                        features_tmp = min_max_scaler.fit_transform(features[view])
                        n_features.append(features_tmp * (weights[view]))
                    else:
                        features_tmp = features[view]
                        n_features.append(features_tmp * (weights[view]))
                z = hstack(n_features)

                kmean = KMeans(n_clusters=self.n_clusters, n_init=10)
                # kmean = KMeans(n_clusters=self.n_clusters, n_init=500, init='random', max_iter=1000, algorithm='full')
                # kmean = KMeans(n_clusters=self.n_clusters, n_init=10, max_iter=1000)
                # kmean = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=1000)
                # kmean = KMeans(n_clusters=self.n_clusters, n_init=500, max_iter=1000, algorithm='full')
                print('Iteration: %d' % (int(ite / update_interval)))
                print("P-step: update {P, C, A} with fixed {Z, U}.")
                y_pred = kmean.fit_predict(z)     # k-means on global features
                
                if initial_flag == 0:
                    y_pred_global = np.copy(y_pred)
                    initial_flag = 1
                
                # print("Update A with fixed C, {Z, U}.")
                new_y, row_ind, col_ind, matrix = self.Match(y_pred, y_pred_global)
                y_pred_global = np.copy(new_y)
                print(matrix)

                # print(kmeans.cluster_centers_.shape)
                # print(y_pred_global[0:9])
                # print(y_pred[0:9])

                acc = np.round(Nmetrics.acc(y[view], y_pred), 5)
                nmi = np.round(Nmetrics.nmi(y[view], y_pred), 5)
                ari = np.round(Nmetrics.ari(y[view], y_pred), 5)
                from Nmetrics import test
                test(y[view], y_pred)
                ACC.append(acc)
                NMI.append(nmi)
                ARI.append(ari)
                # print(kl_loss)
                # print(Loss)
                # print(np.sum(kl_loss), np.sum(Loss))
                print("Z-step: update {Z, U} with fixed {P, C, A}.")
                if y is not None:
                    tmpACC = []
                    tmpNMI = []
                    tmpARI = []
                    for view in range(len(x)):
                        acc = np.round(Nmetrics.acc(y[view], y_pred_sp[view]), 5)
                        nmi = np.round(Nmetrics.nmi(y[view], y_pred_sp[view]), 5)
                        ari = np.round(Nmetrics.ari(y[view], y_pred_sp[view]), 5)
                        from Nmetrics import test
                        test(y[view], y_pred_sp[view])
                        tmpACC.append(acc)
                        tmpNMI.append(nmi)
                        tmpARI.append(ari)
                    vACC.append(tmpACC)
                    vNMI.append(tmpNMI)
                    vARI.append(tmpARI)

                Center_init = kmean.cluster_centers_    # k-means on global features
                new_P = self.new_P(z, Center_init)      # similarity measure
                p = self.target_distribution(new_P)     # enhance
                # p = np.dot(p, matrix)                 # P = E(S(H, C))A,     adjust the arrangement of S
                p = np.dot(p, matrix.T)                 # P = E(S(H, C))A,     arrangement of S is aligned with last iteration
                P = []
                # unify P of supervision loss
                for view in range(len(x)):
                    P.append(p)

                # evaluate the clustering performance
                for view in range(len(x)):
                    Loss[view] = 0.
                    kl_loss[view] = 0.

            # train on batch
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x[0].shape[0])]
            x_batch = []
            y_batch = []
            for view in range(len(x)):
                x_batch.append(x[view][idx])
                y_batch.append(P[view][idx])
                y_batch.append(x[view][idx])
            tmp = self.train_on_batch(xin=x_batch, yout=y_batch)  # [sum, q, xn, q, x]
            # print(tmp)
            KLLoss = []
            for view in range(len(x)):
                Loss[view] += tmp[2 * view + 2]       # lr
                kl_loss[view] += tmp[2 * view + 1]    # lc
                KLLoss.append(tmp[2 * view + 1])
            # MVKLLoss.append(KLLoss)
            MVKLLoss.append(tmp[0])
            index = index + 1 if (index + 1) * batch_size <= x[0].shape[0] else 0
            # print(ite)
            ite += 1
            if ite >= int(maxiter):
                break

        # save the trained model
        # logfile.close()
        print('Saving model to:', save_dir + '/model_final.h5')
        self.model.save_weights(save_dir + '/model_final.h5')
        # self.autoencoder.save_weights(save_dir + '/pre_model.h5')
        # np.save(save_dir + '/AccNmiAriRate/ACC.npy', ACC)
        # np.save(save_dir + '/AccNmiAriRate/NMI.npy', NMI)
        # np.save(save_dir + '/AccNmiAriRate/ARI.npy', ARI)
        # np.save(save_dir + '/AccNmiAriRate/vACC.npy', vACC)
        # np.save(save_dir + '/AccNmiAriRate/vNMI.npy', vNMI)
        # np.save(save_dir + '/AccNmiAriRate/vARI.npy', vARI)
        # np.save(save_dir + '/AccNmiAriRate/TotalLoss.npy', MVKLLoss)
        print('Clustering time: %ds' % (time() - t1))
        print('End clustering:', '-' * 60)

        Q_and_X = self.model.predict(input_dic)
        y_pred = []
        y_softlabels = []
        for view in range(len(x)):
            y_pred.append(Q_and_X[view*2].argmax(1))
            y_softlabels.append(Q_and_X[view*2])

        y_q = Q_and_X[(len(x) - 1) * 2]
        for view in range(len(x) - 1):
            y_q += Q_and_X[view * 2]
        # y_q = y_q/len(x)
        y_mean_pred = y_q.argmax(1)
        return y_pred, y_mean_pred, y_softlabels

    def test_fit(self, arg, x, y, maxiter=2e4, batch_size=256, tol=1e-3,
                UpdateCoo=200, save_dir='./results/tmp'):
        input_dic = {}
        for view in range(len(x)):
            input_dic.update({'input' + str(view + 1): x[view]})
        Q_and_X = self.model.predict(input_dic)
        y_pred = []
        y_softlabels = []
        for view in range(len(x)):
            y_pred.append(Q_and_X[view * 2].argmax(1))
            y_softlabels.append(Q_and_X[view * 2])

        y_q = Q_and_X[(len(x) - 1) * 2]
        for view in range(len(x) - 1):
            y_q += Q_and_X[view * 2]
        # y_q = y_q/len(x)
        y_mean_pred = y_q.argmax(1)
        return y_pred, y_mean_pred, y_softlabels, self.encoder.predict(input_dic)

    def new_P(self, inputs, centers):
        alpha = 1
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(inputs, axis=1) - centers), axis=2) / alpha))
        q **= (alpha + 1.0) / 2.0
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
        return q

    def Match(self, y_true, y_pred):
        # y_modified = Match(y_modified_before, y_modified_target)
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        new_y = np.zeros(y_true.shape[0])

        matrix = np.zeros((D, D), dtype=np.int64)
        matrix[row_ind, col_ind] = 1
        for i in range(y_pred.size):
            for j in row_ind:
                if y_true[i] == col_ind[j]:
                    new_y[i] = row_ind[j]
        return new_y, row_ind, col_ind, matrix
