import tensorflow as tf
from scipy.io import loadmat

import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping# -*- coding: utf-8 -*-

def get_svhn_perceptron_model(shape):
    model = Sequential([
        layers.Flatten(input_shape = shape),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512,  activation='relu'),
        layers.Dense(128,  activation='relu'),
        layers.Dense(10,  activation='softmax'),
    ])
    return model

def get_cnn_model(shape):
    model = Sequential([
        layers.Conv2D(filters=8, input_shape=shape, kernel_size=(5,5), activation='relu', padding='SAME'),
        layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu', padding='SAME'),
        layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='SAME'),
        layers.MaxPooling2D(pool_size=(3,3)),
        layers.Flatten(),
        layers.Dense(32,  activation='relu'),
        layers.Dense(32,  activation='relu'),
        layers.Dense(10,  activation='softmax'),
    ])
    return model

train = loadmat('train_32x32.mat')


train_data = train['X']
train_data = np.transpose(train_data, [3, 0, 1, 2])
train_labels = train['y']
train_labels = np.where(train_labels==10, 0, train_labels) 

test = loadmat('test_32x32.mat')


test_data = test['X']
test_data = np.transpose(test_data, [3, 0, 1, 2])
test_labels = test['y']
test_labels = np.where(test_labels==10, 0, test_labels) 

#pick ten random indices 
ids = random.sample(range(train_data.shape[0]), 10)

#display the images corresponding to ids
images_for_display = train_data[ids,:,:,:]
labels_for_display = train_labels[ids]

f, axarr = plt.subplots(2,5)
plt.axis('off')
for x in range(2):
    for y in range(5):
        idx = x+2*y
        axarr[x,y].imshow(images_for_display[idx,:,:,:])
        axarr[x,y].set_title(labels_for_display[idx][0])
        axarr[x,y].axis('off')
        
model = get_cnn_model(train_data[0,:,:,:].shape)        
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, validation_split=0.05, epochs=30)    