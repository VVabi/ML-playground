# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:58:56 2022

@author: User
"""
import tensorflow as tf
import numpy as np
import glob
import imageio
import matplotlib.pyplot as plot

def read_position(directory):
    return tf.convert_to_tensor(np.float32(np.load(bytes.decode(directory.numpy())+"\\piece_data.npy")))

def read_orientation(directory):
    return tf.convert_to_tensor(np.float32(np.load(bytes.decode(directory.numpy())+"\\orientation_data.npy")))

def read_diagram(directory):
    return tf.convert_to_tensor(np.uint8(imageio.imread(bytes.decode(directory.numpy())+"\\diagram.png")))




def transform_dataset(directory):
    X = read_diagram(directory)
    X.set_shape((128, 128, 3))
    Y = read_position(directory)
    Y.set_shape((8, 8))
    Z = read_orientation(directory)
    Z.set_shape(1)
    return X, Y, Z


def correct_order(X, Y, Z):
    return X, {'pieces_outputs': Y, 'orientation_output': Z}


def get_dataset(base_path):
#base_path = "D:\MLData\ChessDiagrams"

    directories  = glob.glob(base_path+"\*")
    num_samples = len(directories)
    
    file_dataset        = tf.data.Dataset.list_files(base_path+"\*")
    #image_dataset       = file_dataset.map(lambda x: tf.py_function(read_diagram, [x], [tf.float32]), num_parallel_calls=1).map(lambda y: y/255)
    #position_dataset    = file_dataset.map(lambda x: tf.py_function(read_position, [x], [tf.float32]), num_parallel_calls=1)
    #orientation_dataset = file_dataset.map(lambda x: tf.py_function(read_orientation, [x], [tf.float32]), num_parallel_calls=1)
    
    #label_dataset       = tf.data.Dataset.zip((position_dataset, orientation_dataset))
    #full_dataset        = tf.data.Dataset.zip((image_dataset, label_dataset))
    full_dataset = file_dataset.map(lambda x: tf.py_function(transform_dataset, [x], [tf.uint8, tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE).map(correct_order,  num_parallel_calls=tf.data.AUTOTUNE)

    return full_dataset, num_samples
    