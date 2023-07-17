from keras.models import Sequential
from keras.layers import (Dense, Flatten)
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.utils import to_categorical

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf



#Load data
(X_train,y_train), (X_test,y_test) = mnist.load_data()

#Use one-hot encoding for the labels
num_classes = np.unique(y_train).shape[0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test,num_classes)

#Build the model
model = Sequential()
model.add(Flatten(input_shape = X_train[0].shape))
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation="softmax"))
model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

history = model.fit(X_train, y_train,batch_size=128, epochs=100)
print(model.output_shape)
print(X_train[0].shape)


