import h5py
import numpy as np
from PIL import Image
import logging

import numpy as np
import os
import cv2 as cv
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Reshape, GlobalMaxPooling2D, Activation, Dense)
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
from tensorflow.keras import Model, Input
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

import timeit
import datetime
import math

from tensorflow.python.framework.random_seed import set_seed

classes = [
    [1, 23, 31],
    [2, 34, 40],  # !!!
    [3, 41, 47],
    [4, 48, 54],
    [5, 55, 61],
    [6, 62, 68],
    [7, 69, 75],
    [8, 76, 83],
    [9, 84, 90],
    [10, 91, 97],
    [11, 98, 105],
    [12, 106, 157],  # !!!
    [13, 175, 181],
    [14, 182, 188],
    [15, 189, 195],
    [16, 198, 204],
    [17, 205, 211],
    [18, 212, 218],
    [19, 230, 250],  # !!!
]

'''
def evaluate_model_for_task_2(model):
    test_acc = accuracy_score(train_labels, np.argmax(model.predict(train_images), axis=-1))
    print(f"Качество: {test_acc}")
    print("Тест на качество {}".format("не пройден :(" if 0.88 > test_acc else "пройден :)"))
'''

def build_model():
    classification_model = Sequential()
    classification_model.add(
        Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1), input_shape=(23, 5, 3), activation='relu'))
    classification_model.add(BatchNormalization())
    classification_model.add(MaxPooling2D(pool_size=(2, 2)))

    classification_model.add(
        Conv2D(64, kernel_size=(3, 3), padding='same', strides=(1, 1), input_shape=(11, 2, 64), activation='relu'))
    classification_model.add(BatchNormalization())
    classification_model.add(MaxPooling2D(pool_size=(2, 2)))

    classification_model.add(Flatten(name='flatten'))
    classification_model.add(Dense(128, activation='relu'))
    classification_model.add(Dense(20, activation='softmax'))

    classification_model.compile(optimizer='adam',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])
    classification_model.summary()

    return classification_model


def load_images_from_folder_all_info(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            aio_plant = filename.split("_")
            flowering_time = aio_plant[2].split(".")[0]
            for i in classes:
                if int(flowering_time) in range(i[1], i[2] + 1):
                    aio_plant.append(np.uint8(i[0]))
                    aio_plant.append(np.asarray(img))
            images.append(np.asarray(aio_plant))
    return np.asarray(images)


def load_images_from_folder(folder):
    labels = []
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            aio_plant = filename.split("_")
            flowering_time = aio_plant[2].split(".")[0]
            for i in classes:
                if int(flowering_time) in range(i[1], i[2] + 1):
                    labels.append(np.uint8(i[0]))
                    images.append(np.asarray(img).astype(np.float32))

    return np.asarray(images), np.asarray(labels)


def train():
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    K.set_image_data_format('channels_last')

    train_images, train_labels = load_images_from_folder('/Users/mariia/PycharmProjects/vigna/AIO_all')
    # train_images, train_labels = data[:, 3], data[:, 2]

    train_images = train_images / 255.0
    train_vectors = to_categorical(train_labels, 20)

    model = build_model()
    history = model.fit(train_images, train_vectors, epochs=100, batch_size=64, validation_split=0.3)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(100)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Точность на обучении')
    plt.plot(epochs_range, val_acc, label='Точность на валидации')
    plt.legend(loc='lower right')
    plt.title('Точность')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Потери на обучении')
    plt.plot(epochs_range, val_loss, label='Потери на валидации')
    plt.legend(loc='upper right')
    plt.title('Потери')
    plt.savefig('./all_vigna.png')
    plt.show()

    # evaluate_model_for_task_2(classification_model)
    # test_acc = accuracy_score(train_labels, np.argmax(model.predict(train_images), axis=-1))
    # acc = classification_model.predict(train_images)
