# -*- coding: utf-8 -*-

from chessDiagramDataPreparation import get_dataset
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import tensorflow as tf



#Y = np.load('D:\code\ML-playground\chessDiagrams\chess_labels.npy')
#Z = np.load('D:\code\ML-playground\chessDiagrams\chess_orientations.npy')
#X = np.load('D:\code\ML-playground\chessDiagrams\chess_image_data.npy')
#Y = np.flip(Y, axis=1)
#print(X.shape)
#print(Y.shape)

#X_test = X[98000:, :, : ,:]
#Y_test = Y[98000:, :, :]
#X_train = X[:98000,:, :, :]
#Y_train = Y[:98000, :, :]
#Z_test = Z[98000:, :]
#Z_train = Z[:98000,:]


def fix_shape(X, O): #no real clue why this is necessary...
    X.set_shape([None, 128, 128, 3])
    O["pieces_outputs"].set_shape([None, 8, 8]) 
    O["orientation_output"].set_shape([None, 1])
    return X, O


inputs  = keras.Input(shape=(128, 128, 3))
x = layers.Conv2D(filters=8, input_shape=(128, 128, 3),kernel_size=(5,5), activation='relu', padding='SAME')(inputs)
x = layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu', padding='SAME')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='SAME')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), strides = (2,2), activation='relu', padding='SAME')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), strides = (2,2), activation='relu', padding='SAME')(x)
x = layers.Conv2D(filters=64, kernel_size=(3,3), strides = (2,2), activation='relu', padding='SAME')(x)
x = layers.Dropout(0.5)(x)
y = layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu')(x)
y = layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu')(y)
y = layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu')(y)
pieces_outputs =  layers.Conv2D(filters=13, kernel_size=(1,1), activation='softmax', name='pieces_outputs')(y)

z = layers.Flatten()(x)
z = layers.Dense(32, activation='relu')(z)
z = layers.Dense(16, activation='relu')(z)                  
orientation_output = layers.Dense(2, activation='softmax', name = 'orientation_output')(z)                    

model = tf.keras.Model(inputs=inputs,outputs=[pieces_outputs, orientation_output])
      
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss={'pieces_outputs': 'sparse_categorical_crossentropy', 'orientation_output': 'sparse_categorical_crossentropy'},
            metrics={'pieces_outputs':'accuracy',
          'orientation_output':'accuracy'})
                
model.summary()
base_path = "D:\MLData\ChessDiagrams"
dataset, num_entries = get_dataset(base_path)
batch_size = 32
dataset = dataset.shuffle(50).batch(batch_size).map(fix_shape, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(100)

num_test_data = int(0.05*num_entries/batch_size)
num_valid_data = int(0.05*num_entries/batch_size)

test_data = dataset.take(num_test_data)

train_data = dataset.skip(num_test_data)
valid_data = train_data.take(num_valid_data)
train_data = train_data.skip(num_valid_data)

#train_data = dataset.take(int(9500/32))
#valid_data = dataset.skip(int(9500/32))

model.fit(train_data, validation_data=valid_data, epochs=3)
model.learning_rate = 0.0000001
model.fit(train_data, validation_data=valid_data, epochs=100)