import scipy.io as io
import scipy.fft as fft
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def forwardPropagationSigmoid(xvalue):
    sig = (1/(1+np.exp(-xvalue)))
    return sig


def Sigmoid(z):
    if z < 0:
        return 1 - 1/(1 + np.exp(z))
    else:
        return 1/(1 + np.exp(-z))


def assessment(y1, y2):
    m = y1.shape[1]
    lossWB = (-1/m)*np.sum(np.multiply(y1, np.log(y2+0.1)) + np.multiply((1-y1), np.log(1-y2+0.1)))
    return lossWB


def gradientWeight(x, y1, y2):
    gradientW = np.dot(x, (y2 - y1).transpose()) / (y1.shape[0]*y1.shape[1])
    return gradientW


def gradientB(x, y1, y2):
    gradient = ((y2 - y1).sum() / (y1.shape[0]*y1.shape[1]))
    return gradient


def updatingWeights(x, y1, y2, w, b):
    learningRate = 0.4
    updatedW = w - learningRate * gradientWeight(x, y1, y2)
    updatedB = b - learningRate * gradientB(x, y1, y2)
    return updatedW, updatedB


def loadData(filename):
    data = io.loadmat(filename)
    print('Length: ' + str(len(data)))
    print(data.keys())
    print(data['__globals__'])
    data_only = data["data"]
    label = data["label"]
    dataForTraining = np.array(data_only)
    groundTruth = np.array(label)
    newData = dataForTraining.reshape((dataForTraining.shape[0] * dataForTraining.shape[1]), dataForTraining.shape[2])
    train_pct_index = int(0.8 * newData.shape[1])
    X_train, X_test = newData[:, :train_pct_index], newData[:, train_pct_index:]
    Y_train, Y_test = groundTruth[:, :train_pct_index], groundTruth[:, train_pct_index:]
    return X_train, X_test, Y_train, Y_test


def assignWB(x, y):
    w = np.full((x.shape[0], 1), 0.15)
    b = np.full((1, y.shape[1]), 0)
    return w, b


def preProcessing(x, y):
    preX = preprocessing.maxabs_scale(x)
    groundTruth = preprocessing.maxabs_scale(y)
    return preX, groundTruth


def trainModel(x, y):
    w, b = assignWB(x, y)
    print(w.shape, x.shape)
    lossAll = []
    for i in range(0, 1000):
        calc = np.dot(w.transpose(), x) + b
        yCalc = forwardPropagationSigmoid(calc)
        loss = assessment(y, yCalc)
        lossAll.append(loss)
        if loss <= 0.0001:
            break
        w, b = updatingWeights(x, y, yCalc, w, b)
    plt.plot(np.array(lossAll))
    plt.show()
    return w, b


# Testing the data
def testingData(x, y, w, b):
    #w = np.full((X_test.shape[0], 1), )
    b = np.full((1, y.shape[1]), b[0][0])
    xTest, y = preProcessing(x, y)
    x = np.real(fft.fft2(xTest))
    calc = np.dot(w.transpose(), x) + b
    yPred = forwardPropagationSigmoid(calc)
    y01 = np.where(yPred.reshape(-1) > 0.5, 1, 0)
    loss2 = assessment(y, y01)
    print(y01, y01.dtype, y01.shape)
    print('loss after update: ', loss2)
    print(confusion_matrix(y.reshape(-1), y01))
    print(classification_report(y.reshape(-1), y01))


x_train, x_test, y_train, y_test = loadData('WLDataCW.mat')
# Training the data
xTrain, yTrain = preProcessing(x_train, y_train)
trainedW, trainedB = trainModel(xTrain, yTrain)

# Testing the data
xTest, yTest = preProcessing(x_test, y_test)
testingData(xTest, yTest, trainedW, trainedB)


