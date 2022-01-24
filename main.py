from Load_data import load_data_conv
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from sklearn.manifold import TSNE
import os
from time import time
import Nmetrics
from DIMVC import MvDEC
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Layer, InputSpec, Input, Dense, Multiply, concatenate


def _make_data_and_model(args, missrate):
    # prepare dataset
    x, y, size, index = load_data_conv(args.dataset, missrate=missrate)
    view = len(x)
    view_shapes = []
    Loss = []
    Loss_weights = []
    shap_max = 0
    for v in range(view):
        view_shapes.append(x[v].shape[1:])
        if shap_max < x[v].shape[1:][0]:
            shap_max = x[v].shape[1:][0]
    print(shap_max)
    for v in range(view):
        Loss.append('categorical_crossentropy')
        Loss.append('mse')
        Loss_weights.append(args.lc)
        Loss_weights.append(args.Idec)
    print(view_shapes)
    print(Loss)
    print(Loss_weights)
    # prepare optimizer
    optimizer = Adam(lr=args.lr)
    # prepare the model
    n_clusters = len(np.unique(y[0]))
    print("n_clusters:" + str(n_clusters))

    model = MvDEC(n_clusters=n_clusters, view_shape=view_shapes, data=args.dataset)

    model.compile(optimizer=optimizer, loss=Loss, loss_weights=Loss_weights)
    return x, y, model, size, index


def train(args):
    # get data and model
    missrate = args.missrate
    x, y, model, size, index_data = _make_data_and_model(args, missrate=missrate)
    model.model.summary()
    # pretraining
    t0 = time()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # if args.pretrain_dir is not None and os.path.exists(args.pretrain_dir):  # load pretrained weights
    if not args.pretrain:
        model.autoencoder.load_weights(args.pretrain_dir)
        # model.load_weights(args.pretrain_dir)
    else:  # train
        optimizer = Adam(lr=args.lr)
        model.pretrain(x, y, optimizer=optimizer, epochs=args.pretrain_epochs,
                            batch_size=args.batch_size, save_dir=args.save_dir, verbose=0)
        args.pretrain_dir = args.save_dir + '/ae_weights.h5'
    t1 = time()
    print("Time for pretraining: %ds" % (t1 - t0))

    # clustering
    print('Data size:' + str(size))
    view_num = len(x)
    index = np.linspace(0, (1-missrate)*size-1, num=int((1-missrate)*size), dtype=int)
    args.centerinit = 0
    args.maxAR = 0
    # print(index_data)
    for i in ['DIMVC', 'TEST']:
        print(args.maxAR)
        if i == 'DIMVC':
            x_t = []
            y_t = []
            for v in range(len(x)):
                x_t.append(x[v][index])
                y_t.append(y[v][index])
            y_pred, y_mean_pred, _ = model.new_fit(arg=args, x=x_t, y=y_t, maxiter=args.maxiter,
                                                batch_size=args.batch_size, UpdateCoo=args.UpdateCoo,
                                                save_dir=args.save_dir)
        else:
            y_pred, y_mean_pred, y_softlabels, z = model.test_fit(arg=args, x=x, y=y, maxiter=args.maxiter,
                                                       batch_size=args.batch_size, UpdateCoo=args.UpdateCoo,
                                                       save_dir=args.save_dir)
    y_prediction = []
    y_true = []
    y_pre_nomean = []
    y_ture_nomean = []
    if y is not None:
        for view in range(len(x)):
            print(len(y_pred[view]))
            Nmetrics.test(y[view], y_pred[view])   # each view
            y_prediction = y_prediction + list(y_pred[view][int((1-missrate)*size):])
            y_true = y_true + list(y[view][int((1-missrate)*size):])

            y_pre_nomean = y_pre_nomean + list(y_pred[view])
            y_ture_nomean = y_ture_nomean + list(y[view])

        y_prediction = y_prediction + list(y_mean_pred[index])
        y_true = y_true + list(y[0][index])
        print(len(y_prediction))
        Nmetrics.test(np.array(y_true), np.array(y_prediction))  # com mean, incom no mean
        print(len(y_pre_nomean))
        Nmetrics.test(np.array(y_ture_nomean), np.array(y_pre_nomean))  # no mean
    # print(y_prediction)
    true_labels = np.zeros((size, ))
    # print(index_data)
    # print(y)
    for i in range(len(y[0])):
        for v in range(view_num):
            true_labels[index_data[v][i]] = y[v][i]
    # print(true_labels)
    pre_soft_labels = []
    n_clusters = len(np.unique(y[0]))
    for v in range(view_num):
        pre_soft_labels.append(np.zeros((size, n_clusters)))
    # print(pre_soft_labels)
    for i in range(y_softlabels[0].shape[0]):
        for v in range(view_num):
            pre_soft_labels[v][index_data[v][i]] = y_softlabels[v][i]
    # print(pre_soft_labels)

    y_q = np.copy(pre_soft_labels[view_num-1])
    for v in range(view_num-1):
        y_q += pre_soft_labels[v]
    # print(pre_soft_labels)
    # print(y_q)
    y_mean_pred = y_q.argmax(1)
    # print(y_mean_pred)
    t2 = time()
    print("Time for pretaining, clustering and total: (%ds, %ds, %ds)" % (t1 - t0, t2 - t1, t2 - t0))
    print(len(y_mean_pred))
    Nmetrics.test(true_labels, y_mean_pred)     # mean
    print('=' * 60)
    # return Nmetrics.test(true_labels, y_mean_pred)                          # mean
    return Nmetrics.test(np.array(y_true), np.array(y_prediction))            # com mean, incom no mean
    # return Nmetrics.test(np.array(y_ture_nomean), np.array(y_pre_nomean))   # no mean


def test(args):
    assert args.weights is not None
    # x, y, model = _make_data_and_model(args)
    x, y, model, size, index_data = _make_data_and_model(args, missrate=args.missrate)
    model.model.summary()
    print('Begin testing:', '-' * 60)
    model.load_weights(args.weights)
    y_pred, y_mean_pred = model.predict_label(x=x)
    y = y[0]
    if y is not None:
        for view in range(len(x)):
            print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
                    (Nmetrics.acc(y, y_pred[view]), Nmetrics.nmi(y, y_pred[view]), Nmetrics.ari(y, y_pred[view])))
        print('Final: acc=%.4f, nmi=%.4f, ari=%.4f' %
                  (Nmetrics.acc(y, y_mean_pred), Nmetrics.nmi(y, y_mean_pred), Nmetrics.ari(y, y_mean_pred)))
        Nmetrics.test(y, y_mean_pred)
    print('End testing:', '-' * 60)


if __name__ == "__main__":
    # settings
    data = 'Caltech'
    Lc = 1.0
    Lr = 1.0
    lrate = 0.001
    epochs = 500
    Update_epoch = 1000
    Max_iteration = 10
    Batch = 256

    run_times = 1
    results = []
    for missrate in [0.0, 0.1, 0.3, 0.5, 0.7]:
        print("----------------------------missrate-----------------------------------")
        print(missrate)
        print("-----------------------------------------------------------------------")
        import argparse
        parser = argparse.ArgumentParser(description='main')
        parser.add_argument('--dataset', default=data,
                            help="Dataset name to train")
        PATH = './results/'
        path = PATH + data
        train_ae = True
        if train_ae:
            load = None
        else:
            load = path + '/ae_weights.h5'
        TEST = False
        if TEST:
            load_test = path + '/model_final.h5'
        else:
            load_test = None

        parser.add_argument('-d', '--save-dir', default=path,
                            help="Dir to save the model")
        # Parameters for pretraining
        parser.add_argument('--pretrain_dir', default=load, type=str,
                            help="Pretrained weights of the autoencoder")
        parser.add_argument('--pretrain', default=train_ae, type=bool,
                            help="Pretrain the autoencoder?")
        parser.add_argument('--pretrain-epochs', default=epochs, type=int,   # 500
                            help="Number of epochs for pretraining")
        parser.add_argument('-v', '--verbose', default=1, type=int,
                            help="Verbose for pretraining")
        # Parameters for clustering
        parser.add_argument('--testing', default=TEST, type=bool,
                            help="Testing the clustering performance with provided weights")
        parser.add_argument('--weights', default=load_test, type=str,
                            help="Model weights, used for testing")
        parser.add_argument('--lr', default=lrate, type=float,
                            help="learning rate during clustering")
        parser.add_argument('--batch-size', default=Batch, type=int,
                            help="Batch size")
        parser.add_argument('--missrate', default=missrate, type=float,
                            help="Miss rate")
        parser.add_argument('--maxiter', default=Max_iteration * Update_epoch, type=int,
                            help="Maximum number of iterations")
        parser.add_argument('-uc', '--UpdateCoo', default=Update_epoch, type=int,
                            help="Number of iterations to update the target distribution")
        parser.add_argument('--Idec', default=Lr, type=float,
                            help="weight of AEs?")
        parser.add_argument('--lc', default=Lc, type=float,
                            help="weight of clustering")
        args = parser.parse_args()
        print('+' * 30, ' Parameters ', '+' * 30)
        print(args)
        print('+' * 75)
        # testing
        if args.testing:
            test(args)
        else:
            performance = np.zeros(shape=(run_times, 5))
            for i in range(run_times):
                print("---------------------------run_times------------------------------------")
                print(i)
                print("------------------------------------------------------------------------")
                ACC, NMI, V_measure, ARI, Purity = train(args)
                performance[i][0] = ACC
                performance[i][1] = NMI
                performance[i][2] = V_measure
                performance[i][3] = ARI
                performance[i][4] = Purity
            means_per = np.around(np.mean(performance, axis=0), 4)
            results.append(list(means_per))
    # np.save(data + '.npy', results)
    print(results)
