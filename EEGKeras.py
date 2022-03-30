from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, AveragePooling2D, Conv2D
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold, cross_val_predict
from keras import layers, models
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import numpy as np
import scipy.io as io
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

data = io.loadmat('WLDataCW.mat')
X = data['data']
data['label'] = data['label'].reshape(360)

y_convert = pd.get_dummies(data['label'])
X = X.reshape(360, 62*512)

y_convert = np.array(y_convert)
y = y_convert

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

'''
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train)
y_new = np.argmax(y, axis=1)
kfold = KFold(n_splits=5, shuffle=True)
accuracies = cross_val_score(estimator=neigh,
                             X=X,
                             y=y_new,
                             cv=kfold, scoring='accuracy')
print(accuracies.mean())
'''

def make_classifier():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(32, activation='sigmoid'))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(16, activation='sigmoid'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(8, activation='sigmoid'))
    model.add(layers.Dropout(rate=0.2))
    model.add(Flatten())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    return model


kfold = KFold(n_splits=5, shuffle=True)
classifier = KerasClassifier(build_fn=make_classifier, verbose=1)
classifier.fit(X_train, y_train, batch_size=64, epochs=50)
y_new = np.argmax(y, axis=1)
accuracies = cross_val_score(estimator=classifier,
                             X=X,
                             y=y_new,
                             cv=kfold, scoring='accuracy')
print(accuracies.mean())
