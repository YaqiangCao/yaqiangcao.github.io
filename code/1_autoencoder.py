# sys
import os

#3rd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

#3rd deep-learning
from keras.optimizers import Adam
from keras.datasets import mnist  #minist used as test data
from keras.layers import Dense, Dropout
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

#global settins
#global settings for tensorflow
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def getModel(input_shape, act="elu", dim=2, checkpoint=None):
    """
    input_shape: int, the length of input vector
    act: str, the name of normal activation function used in encoder and decodr layers.
    dim: 2, the target dimensions that reduced 
    """
    if os.path.isfile(checkpoint):
        model = load_model(checkpoint)
    else:
        model = Sequential()
        #model.add( Dense(1024,name="encode1",activation=act,input_dim=(input_shape,)) )
        model.add(
            Dense(1024, name="encode1", activation=act, input_dim=input_shape))
        model.add(Dropout(0.1))
        model.add(Dense(512, name="encode2", activation=act))
        model.add(Dense(256, name="encode3", activation=act))
        model.add(Dense(128, name="encode4", activation=act))
        model.add(Dense(dim, name="bottleneck", activation="linear"))
        #model.add(Dense(dim, name="bottleneck", activation=act))
        model.add(Dense(128, name="decode4", activation=act))
        model.add(Dense(256, name="decode3", activation=act))
        model.add(Dense(512, name="decode2", activation=act))
        model.add(Dense(1024, name="decode1", activation=act))
        model.add(Dropout(0.1))
        model.add(Dense(input_shape, name="final", activation="sigmoid"))
        #model.add(Dense(input_shape, name="final", activation="tanh"))
        #mse can be used as if the input is float values, if target is 0,1 binary, binary_crossentropy can be used as well.
        #model.compile(loss="mean_squared_error",optimizer=Adam(),metrics=["mae","mse"])
        model.compile(loss="mean_squared_error", optimizer=Adam())
        #model.compile(loss="binary_crossentropy",optimizer=Adam(),metrics=["mae","mse"])
    print(model.summary())
    cp = ModelCheckpoint(checkpoint,
                         monitor="val_loss",
                         verbose=1,
                         save_weights_only=False,
                         save_best_only=True,
                         mode="min")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=10)
    early_stop = EarlyStopping(monitor="val_loss", patience=30)
    return model, [cp, reduce_lr, early_stop]  #callbacks


def showLoss(f, outf):
    mat = pd.read_csv(f, index_col=0, sep="\t")
    fig, ax = pylab.subplots()
    ax.plot(mat.index, mat["loss"], label="train loss")
    ax.plot(mat.index, mat["val_loss"], label="test loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    pylab.tight_layout()
    pylab.savefig(outf + "_loss.pdf")


def trainMnist():
    #load mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #reshape ,784 is 28*28
    x_train = x_train.reshape(x_train.shape[0], 784) / 255.0
    x_test = x_test.reshape(x_test.shape[0], 784) / 255.0
    model, callbacks = getModel(784, act="elu", checkpoint="mnist.h5")
    hist = model.fit(x_train,
                     x_train,
                     batch_size=256,
                     epochs=500,
                     verbose=1,
                     validation_data=(x_test, x_test),
                     callbacks=callbacks)
    hist = pd.DataFrame(hist.history)
    hist.to_csv("mnist_autoencoder_trainningHistroy.txt",
                sep="\t",
                index_label="epoch")


def showMnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 784) / 255.0
    x_test = x_test.reshape(x_test.shape[0], 784) / 255.0
    model = load_model("mnist.h5")
    encoder = Model(model.input, model.get_layer("bottleneck").output)
    #show the dimension
    fig, axs = pylab.subplots(1, 2)
    rep = encoder.predict(x_train)  
    axs[0].set_title("trainning data")
    axs[0].scatter(rep[:, 0], rep[:, 1], c=y_train, s=8, cmap="tab10")

    rep = encoder.predict(x_test)  
    axs[1].set_title("test data")
    axs[1].scatter(rep[:, 0], rep[:, 1], c=y_test, s=8, cmap="tab10")
    pylab.tight_layout()
    pylab.savefig("mnist_autoencoder.pdf")


def norm10x(mat, zscore=True, minmax=False):
    """
    Normalization according to https://kb.10xgenomics.com/hc/en-us/articles/115004583806-How-are-the-UMI-counts-normalized-before-PCA-and-differential-expression-
    1) normlize count to median per sample
    2) z-score/minmax for each gene
    """
    mat = np.array(mat)
    print(mat.shape)
    #remove 0 genes, 0 samples
    ns = []
    for i in range(mat.shape[0]):
        if np.sum(mat[i, :]) > 10:
            ns.append(i)
    mat = mat[ns, :]
    ns = []
    for j in range(mat.shape[1]):
        if np.sum(mat[:, j]) > 1000:
            ns.append(j)
    mat = mat[:, ns]
    ss = np.sum(mat, axis=0)  #total sequencing number of each cell
    #print(ss.shape)
    ss = np.median(ss, axis=0)
    for i in range(mat.shape[1]):
        sf = np.sum(mat[:, i]) / ss
        mat[:, i] = mat[:, i] / sf
        #print(np.sum(mat[:,i]),ss,sf)
    mat = np.log2(mat + 1.0)
    ns = []
    #remove all same genes,maybe all 0.
    for j in range(mat.shape[0]):
        if mat[j, :].std() > 0.0:
            ns.append(j)
    mat = mat[ns, :]
    for j in range(mat.shape[0]):
        if zscore:
            mat[j, :] = (mat[j, :] -
                         mat[j, :].mean()) / mat[j, :].std()  #zscore
        if minmax:
            mat[j, :] = (mat[j, :] - mat[j, :].min()) / (
                mat[j, :].max() - mat[j, :].min())  #normalized to 0-1
        #print(j,mat[j,:].mean(),mat[j,:].std())
    print(mat.shape)
    #print(np.max(mat),np.min(mat))
    return mat.T  #each row a cell, each column a gene


def train10x():
    #load 10x data
    from get10x import get_matrix_from_h5
    mat = get_matrix_from_h5("neuron_10k_v3_filtered_feature_bc_matrix.h5")
    expMat = mat.matrix.todense()
    #here is to clustering cells, like mnist, make each row a cell
    expMat = norm10x(expMat)
    x_train, x_test = train_test_split(expMat, test_size=0.05)
    for act in ["elu", "relu", "linear", "tanh", "sigmoid"]:
        model, callbacks = getModel(expMat.shape[1],
                                    act=act,
                                    checkpoint="10x_neron_10k_%s.h5" % act)
        hist = model.fit(x_train,
                         x_train,
                         batch_size=256,
                         epochs=500,
                         verbose=1,
                         validation_data=(x_test, x_test),
                         callbacks=callbacks)
        hist = pd.DataFrame(hist.history)
        hist.to_csv("10x_%s_trainningHistroy.txt" % act,
                    sep="\t",
                    index_label="epoch")


def show10x():
    from get10x import get_matrix_from_h5
    mat = get_matrix_from_h5("neuron_10k_v3_filtered_feature_bc_matrix.h5")
    #here is to clustering cells, like mnist, make each row a cell
    expMat = mat.matrix.todense()
    expMat = norm10x(expMat)
    fig, axs = pylab.subplots(1, 5, figsize=(12, 2.75))
    for i, act in enumerate(["elu", "relu", "linear", "tanh", "sigmoid"]):
        model = load_model("10x_neron_10k_%s.h5" % act)
        encoder = Model(model.input, model.get_layer("bottleneck").output)
        rep = encoder.predict(expMat)
        axs[i].scatter(rep[:, 0], rep[:, 1], s=5, cmap="tab10")
        axs[i].set_title(act)
    pylab.tight_layout()
    pylab.savefig("10x.pdf")


#trainMnist()
#showLoss("mnist_autoencoder_trainningHistroy.txt","minist")
#showMnist()
train10x()
show10x()
