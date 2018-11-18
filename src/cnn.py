'''
This script provides functions to build a convolutional neural network using Keras.
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from clean_data import Data
from baseline_models import report_accuracy, plot_roc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import TensorBoard


def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes):
    '''
    INPUT:
    learning_rate, learning-rate for the optimizer
    num_dense_layers, number of dense layers
    num_dense_nodes, number of nodes in each dense layer
    OUTPUT:
    This function creates a convolutional neural network.
    '''
    # Create an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    inputs = Input(shape=(img_size_flat,))

    # Variable used for building the Neural Network.
    net = inputs

    # The input is an image as a flattened array with 784 elements.
    # But the convolutional layers expect images with shape (28, 28, 1)
    net = Reshape(img_shape_full)(net)

    # First convolutional layer with ReLU-activation and max-pooling.
    net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)

    # Second convolutional layer with ReLU-activation and max-pooling.
    net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)

    # Third convolutional layer with ReLU-activation and max-pooling.
    net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv3')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)

    # Flatten the output of the conv-layer from 4-dim to 2-dim.
    net = Flatten()(net)

    for i in range(num_dense_layers):
        net = Dense(num_dense_nodes, activation='relu')(net)

    # Last fully-connected / dense layer with softmax-activation, used for classification.
    net = Dense(num_classes, activation='softmax')(net)

    outputs = net

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(lr=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_data(df, df_test):
    '''
    INPUT:
    df, pandas data frame of your test data
    OUTPUT:
    This function returns the train, validation and test data you need for the CNN model
    '''
    data = Data(df)
    X = data.X
    y = data.y
    rng_seed = 2
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42)

    data_test = Data(df_test)
    X_test = data_test.X
    y_test = data_test.y

    y_train_ohe = np_utils.to_categorical(y_train)
    y_val_ohe = np_utils.to_categorical(y_val)
    validation_data = (X_val, y_val_ohe

    return X_train, X_test, y_train_ohe, y_test, validation_data


def set_data_dimension(img_size = 28):
    '''
    INPUT:
    img_size, number of pixels of your input image in each dimension
    OUTPUT:
    This function returns the data dimension you need for the CNN model
    '''
    img_size_flat = img_size * img_size
    img_shape = (img_size, img_size)
    # Tuple with height, width and depth used to reshape arrays.
    img_shape_full = (img_size, img_size, 1)

    return img_size_flat, img_shape_full


def main():
    df = pd.read_csv('../data/final.csv',index_col=0)
    df_test = pd.read_csv('../data/final_test.csv',index_col=0)
    X_train, X_test, y_train_ohe, y_test, validation_data = get_data(df)
    num_channels = 1
    num_classes = 4
    img_size_flat, img_shape_full = set_data_dimension()
    # best model: 3.0e-04, 5, 231, 100, 20
    # base model: 1.0e-05, 1, 16, 128, 20
    learning_rate, num_dense_layers, num_dense_nodes = 1.0e-05, 1, 16
    model = create_model(learning_rate, num_dense_layers, num_dense_nodes)

    #tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=100, write_graph=True, write_grads=True, write_images=True)
    model.fit(x=X_train,y=y_train_ohe, epochs=20,batch_size=128,validation_data=validation_data) # callbacks = [tensorboard]
    model.save('cnn_best_1.h5')
    y_prod = model.predict(x=X_test)
    y_pred = np.argmax(y_prod, axis=1)
    report_accuracy(y_test,y_pred)
    y_test_binarize = label_binarize(y_test, classes=[0, 1, 2, 3])
    plot_roc(y_test_binarize, y_prod, title = 'Optimized CNN')

if __name__ == '__main__':
    main()
