import numpy as np
import math
import scipy.io as io
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

data = io.loadmat('WLDataCW.mat')
print('Length: ' + str(len(data)))
print(data.keys())
print(data['__globals__'])
data_only = data["data"]
label = data["label"]
print(data_only.shape)
print(label.shape)

X = np.array(data_only)
y = np.array(label)

# Keras Definement
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
