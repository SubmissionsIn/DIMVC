import numpy as np
from keras_preprocessing import image
# from PIL import Image
from numpy import hstack
from scipy import misc
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.preprocessing import normalize
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

path = './data'


def Caltech(missrate=0.5):
    # the data is already be pre-processed
    Data = scio.loadmat(path + "/Caltech.mat")
    x1 = Data['X1']
    x2 = Data['X2']
    Y = Data['Y']
    Y = Y.reshape(Y.shape[0])
    size = Y.shape[0]
    X, Y, index = Form_Incomplete_Data(missrate=missrate, X=[x1, x2], Y=[Y, Y])
    return X, Y, size, index


def Form_Incomplete_Data(missrate=0.5, X = [], Y = []):
    size = len(Y[0])
    view_num = len(X)
    t = np.linspace(0, size - 1, size, dtype=int)
    import random
    random.shuffle(t)
    Xtmp = []
    Ytmp = []
    for i in range(view_num):
        xtmp = np.copy(X[i])
        Xtmp.append(xtmp)
        ytmp = np.copy(Y[i])
        Ytmp.append(ytmp)
    for v in range(view_num):
        for i in range(size):
            Xtmp[v][i] = X[v][t[i]]
            Ytmp[v][i] = Y[v][t[i]]
    X = Xtmp
    Y = Ytmp

    # complete data index
    index0 = np.linspace(0, (1 - missrate) * size - 1, num=int((1 - missrate) * size), dtype=int)
    missindex = np.ones((int(missrate * size), view_num))
    print(missindex.shape)
    # incomplete data index
    index = []
    for i in range(missindex.shape[0]):
        missdata = np.random.randint(0, high=view_num, size=view_num - 1)
        # print(missdata)
        missindex[i, missdata] = 0
    # print(missindex)
    for i in range(view_num):
        index.append([])
    miss_begain = (1 - missrate) * size
    for i in range(missindex.shape[0]):
        for j in range(view_num):
            if missindex[i, j] == 1:
                index[j].append(int(miss_begain + i))
    # print(index)
    maxmissview = 0
    for j in range(view_num):
        if maxmissview < len(index[j]):
            print(len(index[j]))
            maxmissview = len(index[j])
    print(maxmissview)
    # add some incomplete views' data index to equal for convenience
    for j in range(view_num):
        flag = np.random.randint(0, high=size, size=maxmissview - len(index[j]))
        index[j] = index[j] + list(flag)
    # to form complete and incomplete views' data
    for j in range(view_num):
        index[j] = list(index0) + index[j]
        X[j] = X[j][index[j]]
        print(X[j].shape)
        Y[j] = Y[j][index[j]]
        print(Y[j].shape)
    print("----------------generate incomplete multi-view data-----------------------")
    return X, Y, index


def load_data_conv(dataset, missrate):
    print("load:", dataset)
    if dataset == 'Caltech':                # Caltech
        return Caltech(missrate=missrate)
    else:
        raise ValueError('Not defined for loading %s' % dataset)
