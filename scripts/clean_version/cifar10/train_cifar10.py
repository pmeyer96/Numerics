"""Script to train a CIFAR10 neural network and save it to a keras model"""

import numpy as np
from keras.datasets import cifar10
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential, save_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train /= 255
X_test /= 255

num_classes = np.unique(y_train).shape[0]
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
        input_shape=X_train[0].shape,
    )
)
model.add(BatchNormalization())
model.add(
    Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(
    Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )
)
model.add(BatchNormalization())
model.add(
    Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(
    Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )
)
model.add(BatchNormalization())
model.add(
    Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# Augment training data with shifts and flips.
datagen = ImageDataGenerator(
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
)
it_train = datagen.flow(X_train, y_train, batch_size=64)

# Train the model. Note that fit() does not train on the validation data.
history = model.fit(
    it_train,
    epochs=2,
    verbose=True,
    validation_data=(X_test, y_test),
    steps_per_epoch=int(X_train.shape[0] / 64),
)

save_model(model, "/home/patric/Masterthesis/Numerics/data/CIFAR_10/model.keras")
