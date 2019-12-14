

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

def initialize_parameters(shape, name):
    tf.set_random_seed(1)

    return tf.get_variable(name=name,shape=shape, initializer=tf.contrib.layers.xavier_initializer(seed = 0))



def new_conv_layer(X, #input
                   W,
                   pool_size, pool_strides,
                   use_pooling=True, ):

    #Create layer
    layer=tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')


    if use_pooling:
        layer=tf.nn.max_pool(layer, ksize=[1,pool_size,pool_size,1], strides=[1,pool_strides,pool_strides,1], padding='SAME')
    
    layer=tf.nn.relu(layer)

    return layer

def forward_propagation(X):

    #Create the first layer
    W1=initialize_parameters([4,4,3,8], 'W1')
    layer1=new_conv_layer(X,W1,8,8, use_pooling=True)

    #Create the second layer
    W2=initialize_parameters([2,2,8,16], 'W2')
    layer2=new_conv_layer(layer1,W2,4,4, use_pooling=True)

    #Flatten
    F=tf.contrib.layers.flatten(layer2)

    #Fully
    layer3=tf.contrib.layers.fully_connected(F, 6,activation_fn=None ) 


    parameters= {'W1': W1, 'W2':W2}

    return layer3, parameters

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X=tf.placeholder(tf.float32, [None, 64, 64, 3])
    Y=tf.placeholder(tf.float32, [None, 6])
    Z3 = forward_propagation(X)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
    print("Z3 = \n" + str(a))

def compute_cost(layer3, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer3, labels=y))

def model(X_train, y_train, X_test, y_test, learnning_rate=0.009, num_epochs=100, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed=3
    (m, n_H0, n_W0, n_C0)=X_train.shape
    n_y=y_train.shape[1]

    X=tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    y=tf.placeholder(tf.float32, [None, n_y])


    layer3, parameters=forward_propagation(X)
    cost=compute_cost(layer3, y)

    #Optimizer

    optimizer=tf.train.AdamOptimizer(learning_rate=learnning_rate).minimize(cost)

    init=tf.global_variables_initializer()
    start_time=time.time()

    with tf.Session() as sess:
        sess.run(init)
        costs=list()
        for epoch in range(num_epochs):
            seed+=1
            num_minibatches=int (m/minibatch_size)
            minibatches=random_mini_batches(X_train, y_train, minibatch_size,seed)
            minibatch_cost=0

            for minibatch in minibatches:
                (minibatch_X, minibatch_y)=minibatch

                _, temp_cost=sess.run(fetches=[optimizer,cost ],
                                      feed_dict={X:minibatch_X,
                                                 y:minibatch_y})
                minibatch_cost+= temp_cost/num_minibatches
            costs.append(minibatch_cost)
            if epoch %10==0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        #Correct_prediction
        predicts=tf.argmax(layer3, 1)
        corrects=tf.equal(predicts, tf.arg_max(y, 1))

        #Train and test accuracy

        acc=tf.reduce_mean(tf.cast(corrects, 'float'))
        train_acc=sess.run(acc,{X:X_train, y:y_train} )
        test_acc=sess.run(acc,{X:X_test, y:y_test})
        print('Train accuracy', train_acc)
        print('Test accuracy', test_acc)

        
    end_time=time.time()
    time_diff=end_time-start_time

    print('Time usage: '+ str(timedelta(seconds=int(round(time_diff)))))

    

    return train_acc, test_acc, parameters

def plot_images(images):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])

X_train_org, y_train_org, X_test_org, y_test_org, classes=load_dataset()

X_train=X_train_org/255.
X_test=X_test_org/255.
y_train=convert_to_one_hot(y_train_org,6).T
y_test=convert_to_one_hot(y_test_org,6).T

plot_images(X_train_org)

train_acc, test_acc, parameters=model(X_train, y_train, X_test, y_test)

