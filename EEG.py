import math

import scipy.io as io
import scipy.fft as fft
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score


# Forward Propagation by Sigmoid function
def forwardPropagationSigmoid(xvalue):
    sig = (1/(1+np.exp(-xvalue)))
    return sig


def Sigmoid(z):
    if z < 0:
        return 1 - 1/(1 + math.exp(z))
    else:
        return 1/(1 + math.exp(-z))


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


def updatingWeights(x, y1, y2):
    learningRate = 0.81
    updatedW = w - learningRate * gradientWeight(x, y1, y2)
    updatedB = b - learningRate * gradientB(x, y1, y2)
    return updatedW, updatedB


data = io.loadmat('WLDataCW.mat')
print('Length: ' + str(len(data)))
print(data.keys())
print(data['__globals__'])
data_only = data["data"]
label = data["label"]
print(data_only.shape)
print(label.shape)

dataForTraining = np.array(data_only)
groundTruth = np.array(label)

newData = dataForTraining.reshape((dataForTraining.shape[0] * dataForTraining.shape[1]), dataForTraining.shape[2])

train_pct_index = int(0.8 * newData.shape[1])
X_train, X_test = newData[:, :train_pct_index], newData[:, train_pct_index:]
y_train, y_test = groundTruth[:, :train_pct_index], groundTruth[:, train_pct_index:]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

'''
print(newData.transpose().shape)
print(groundTruth.transpose().shape)
x_train, x_test, y_train, y_test = train_test_split(newData, groundTruth, test_size=0.20)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
'''
'''
kfold = KFold(5)

for train, test in kfold.split(newData):
    print('train: %s, test: %s' % (newData[train], newData[test]))
    # print(newData[train].shape, newData[test].shape)
'''

w = np.full((X_train.shape[0], 1), 0.15)
b = np.full((1, y_train.shape[1]), 0)

print(w, w.shape)
print(b, b.shape)

y = preprocessing.maxabs_scale(X_train)
groundTruth = preprocessing.maxabs_scale(y_train)
print(groundTruth)

x = np.real(fft.fft2(y))
#y = newData
#print(x, x.shape, x.dtype)

#y = preprocessing.MinMaxScaler().fit_transform(newData)

#X_train, X_test, Y_train, Y_test = train_test_split(y, groundTruth, test_size=0.25)


'''
calc = np.dot(w.transpose(), y) + b
yCalc = forwardPropagationSigmoid(calc)
print(yCalc)
print(yCalc.shape)
loss = assessment(groundTruth, yCalc)
print(loss)

w, b = updatingWeights(y, groundTruth, yCalc)
print(w, w.shape)
print(b, b.shape)


nData = preprocessing.MinMaxScaler().fit_transform(newData)
nDataW = preprocessing.MinMaxScaler().fit_transform(w)
# calc = np.dot(w.transpose(), newData[train]) + b
calc = np.dot(w.transpose(), y) + b
yCalc = forwardPropagationSigmoid(calc)
loss2 = assessment(groundTruth, yCalc)
print('loss after update: ', loss2)
'''

lossAll = []
dAll = [w]

for i in range(0, 100):
    calc = np.dot(w.transpose(), y) + b
    yCalc = forwardPropagationSigmoid(calc)
    print(yCalc.shape, yCalc.dtype)
    #print(yCalc)
    #print(yCalc.shape)
    loss = assessment(y_train, yCalc)
    lossAll.append(loss)
    w, b = updatingWeights(y, y_train, yCalc)
    #print(w, w.shape)
    #print(b, b.shape)

    #y_pred = log.predict(y)
    print(yCalc.shape, yCalc.dtype)
    print(groundTruth.shape, groundTruth.dtype)

'''
print(np.round(yCalc))
print(groundTruth)
score = accuracy_score(groundTruth, yCalc)
print('Score :', score)
'''

print(lossAll)
print(len(lossAll))
plt.plot(np.array(lossAll))
plt.show()
# print(newData.shape)
# Testing the data
w = np.full((X_test.shape[0], 1), 0.15)
b = np.full((1, y_test.shape[1]), 0)

print(w, w.shape)
print(b, b.shape)

xTest = preprocessing.maxabs_scale(X_test)
yTrue = preprocessing.maxabs_scale(y_test)
print(yTrue)

x = np.real(fft.fft2(xTest))
calc = np.dot(w.transpose(), x) + b
yPred = forwardPropagationSigmoid(calc)
loss2 = assessment(yTrue, yPred)
print('loss after update: ', loss2)
score = accuracy_score(yTrue, yPred)
print('Accuracy :', score)
