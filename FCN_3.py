
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, callbacks
import numpy as np
import tensorflow as tf


def FCN_Net():
    model = tf.keras.Sequential([
        layers.Input(shape=(800, 800, 3,)),

        layers.Conv2D(16, kernel_size=[3, 3], padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid'),
        # [400, 400]

        layers.Conv2D(8, kernel_size=[3, 3], padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid'),
        # [200, 200]
        layers.Conv2D(8, kernel_size=[3, 3], padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid'),
        # [100, 100]
        layers.Conv2D(8, kernel_size=[3, 3], padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid'),
        # [50, 50]
        layers.Conv2D(8, kernel_size=[3, 3], padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid'),
        # [25, 25]
        # upsampling
        layers.Conv2DTranspose(8, kernel_size=[2, 2], strides=2, padding='same',activation='relu'),
        # [50, 50]
        layers.Conv2D(8, kernel_size=[3, 3], padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        # [50, 50]
        # upsampling
        layers.Conv2DTranspose(8, kernel_size=[2, 2], strides=2, padding='same',activation='relu'),
        # [100, 100]
        layers.Conv2D(8, kernel_size=[3, 3], padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        # [100, 100]
        # upsampling
        layers.Conv2DTranspose(8, kernel_size=[2, 2], strides=2, padding='same',activation='relu'),
        # [200, 200]
        layers.Conv2D(8, kernel_size=[3, 3], padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        # [200, 200]
        # upsampling
        layers.Conv2DTranspose(8, kernel_size=[2, 2], strides=2, padding='same',activation='relu'),
        # [400, 400]
        layers.Conv2D(8, kernel_size=[3, 3], padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        # [400, 400]
        layers.Conv2DTranspose(8, kernel_size=[2, 2], strides=2, padding='same',activation='relu'),
        # [800, 800]
        layers.Conv2D(2, kernel_size=[3, 3], padding='same'),
    ])

    return model


