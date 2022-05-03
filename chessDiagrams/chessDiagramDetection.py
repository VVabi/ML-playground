# -*- coding: utf-8 -*-

from diagramGenerator import getPieceDict, createChessDiagramData
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import tensorflow as tf
import chess
import chess.svg

def create_data(): #TODO make this more configurable
    fens = []
    
    with open('/home/vabi/code/chessEngine/quiet-labeled.epd', 'r') as f:
        fens = list(f)
        

    num_samples = 100000
    offset = 40000
    print(num_samples)  
    print("\n")
    offset = 0
    X = np.zeros((num_samples, 128, 128, 3), dtype='uint8')
    Y = np.zeros((num_samples, 8, 8))
    Z = np.zeros((num_samples, 1))
    piece_dict= getPieceDict()
    
    for ind in range(num_samples):
        if ind % 100 == 0:
            print("\r", end="")
            print(ind/num_samples, end="     ")
        fen = fens[ind+offset]
        parts = fen.split(' ')
        
        next_fen = parts[0]+' '+parts[1]+' ' + parts[2] + ' - 0 1'
        color = chess.WHITE
        if np.random.rand(1)[0] < 0.5:
            color = chess.BLACK
        loc_X, loc_Y, loc_Z = createChessDiagramData(next_fen, piece_dict, color)
        X[ind,:,:,:] = loc_X
        Y[ind,:,:] = loc_Y
        Z[ind, :] = loc_Z
    
    np.save('chess_image_data_extra', X, allow_pickle=False)
    np.save('chess_labels_extra', Y, allow_pickle=False)
    np.save('chess_orientations_extra', Z, allow_pickle=False)

def get_prediction(S):
    Y= np.zeros((8, 8))
    
    for x in range(8):
        for y in range(8):
            Y[x, y] = np.argmax(S[x, y, :])
    
    return Y


X = np.load('chess_image_data.npy')
Y = np.load('chess_labels.npy')
Z = np.load('chess_orientations.npy')
#Y = np.flip(Y, axis=1)
print(X.shape)
print(Y.shape)

X_test = X[9000:, :, : ,:]
Y_test = Y[9000:, :, :]
X_train = X[:9000,:, :, :]
Y_train = Y[:9000, :, :]
Z_test = Z[9000:, :]
Z_train = Z[:9000,:]

inputs  = keras.Input(shape=(128, 128, 3))
x = layers.Conv2D(filters=8, input_shape=(128, 128, 3),kernel_size=(5,5), activation='relu', padding='SAME')(inputs)
x = layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu', padding='SAME')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='SAME')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), strides = (2,2), activation='relu', padding='SAME')(x)
x = layers.Conv2D(filters=32, kernel_size=(3,3), strides = (2,2), activation='relu', padding='SAME')(x)
x = layers.Conv2D(filters=64, kernel_size=(3,3), strides = (2,2), activation='relu', padding='SAME')(x)
y = layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu')(x)
y = layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu')(y)
y = layers.Conv2D(filters=32, kernel_size=(1,1), activation='relu')(y)
pieces_outputs =  layers.Conv2D(filters=13, kernel_size=(1,1), activation='softmax', name='pieces_outputs')(y)

z = layers.Flatten()(x)
z = layers.Dense(32, activation='relu')(z)
z = layers.Dense(16, activation='relu')(z)                  
orientation_output = layers.Dense(2, activation='softmax', name = 'orientation_output')(z)                    

model = tf.keras.Model(inputs=inputs,outputs=[pieces_outputs, orientation_output])
      
model.compile(optimizer='adam',
              loss={'pieces_outputs': 'sparse_categorical_crossentropy', 'orientation_output': 'sparse_categorical_crossentropy'},
            metrics={'pieces_outputs':'accuracy',
          'orientation_output':'accuracy'})
                
model.summary()

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, [Y_train, Z_train], validation_split=0.05, epochs=20)
 
 