import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.metrics import  confusion_matrix
import time 
from datetime import timedelta
import cv2
from tensorflow.python.framework import ops

def load_dataset():
    train_data=h5py.File('datasets/train_signs.h5', 'r')
    train_set_x=np.array(train_data['train_set_x'][:])
    train_set_y=np.array(train_data['train_set_y'][:])


    test_data=h5py.File('datasets/test_signs.h5', 'r')
    test_set_x=np.array(test_data['test_set_x'][:])
    test_set_y=np.array(test_data['test_set_y'][:])
    

    classes=np.array(test_data['list_classes'][:])

    train_set_y=train_set_y.reshape((1,train_set_y.shape[0]))
    test_set_y=test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x, train_set_y, test_set_x, test_set_y, classes

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def random_mini_batches(X, y, mini_batch_size=64, seed=0):

    m=X.shape[0] #number of training examples
    mini_batches=[]
    np.random.seed(seed)

    #Shuffle (X,Y)
    permutation=list(np.random.permutation(m))
    shuffle_X=X[permutation, :,:,:]
    shuffle_y=y[permutation,:]

    #Partition 
    num_of_batches=math.floor(m/mini_batch_size)

    for k in range(num_of_batches):
        mini_batch_X=shuffle_X[k*mini_batch_size: (k+1)*mini_batch_size,:,:,:]
        mini_batch_y=shuffle_y[k*mini_batch_size: (k+1)*mini_batch_size,:]
        mini_batch=(mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    #Handle the last batch

    if m % mini_batch_size !=0:
        mini_batch_X=shuffle_X[num_of_batches*mini_batch_size:,:,:,:]
        mini_batch_y=shuffle_y[num_of_batches*mini_batch_size:,:]
        mini_batch=(mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches