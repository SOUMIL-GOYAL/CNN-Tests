#> cd C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv
import os
from zipfile import ZipFile
import time
from datetime import datetime
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model


# Setting random seeds to reduce the amount of randomness in the neural net weights and results.
# The results may still not be exactly reproducible.
np.random.seed(42)
tf.random.set_seed(42)

tf.__version__

# Importing the training dataset and testing dataset to create tensors of images using the filename paths.


train_df = pd.read_csv("data/images_filenames_labels_train.csv")
test_df = pd.read_csv("data/images_filenames_labels_test.csv")

# Dropping the age column since classes of age-ranges have been re-distributed from 11 to 7 classes.

train_df.drop(columns=['target'], inplace=True)
train_df.head()

# Dropping the age column since classes of age-ranges have been re-distributed from 11 to 7 classes.

test_df.drop(columns=['target'], inplace=True)
test_df.head()

# Defining a function to return the class labels corresponding to the re-distributed 7 age-ranges.

def class_labels_reassign(age):

    if 1 <= age <= 2:
        return 0
    elif 3 <= age <= 9:
        return 1
    elif 10 <= age <= 20:
        return 2
    elif 21 <= age <= 27:
        return 3
    elif 28 <= age <= 45:
        return 4
    elif 46 <= age <= 65:
        return 5
    else:
        return 6
    

train_df['target'] = train_df['age'].map(class_labels_reassign)
test_df['target'] = test_df['age'].map(class_labels_reassign)


train_df['target'].value_counts(normalize=True)


test_df['target'].value_counts(normalize=True)


# Converting the filenames and target class labels into lists for augmented train and test datasets.

train_filenames_list = list(train_df['filename'])
train_labels_list = list(train_df['target'])

test_filenames_list = list(test_df['filename'])
test_labels_list = list(test_df['target'])


# Creating tensorflow constants of filenames and labels for augmented train and test datasets from the lists defined above.

train_filenames_tensor = tf.constant(train_filenames_list)
train_labels_tensor = tf.constant(train_labels_list)

test_filenames_tensor = tf.constant(test_filenames_list)
test_labels_tensor = tf.constant(test_labels_list)


# Defining a function to read the image, decode the image from given tensor and one-hot encode the image label class.
# Changing the channels para in tf.io.decode_jpeg from 3 to 1 changes the output images from RGB coloured to grayscale.

num_classes = 7

def _parse_function(filename, label):
    
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=1)    # channels=1 to convert to grayscale, channels=3 to convert to RGB.
    # image_resized = tf.image.resize(image_decoded, [200, 200])
    label = tf.one_hot(label, num_classes)

    return image_decoded, label



# Getting the dataset ready for the neural network.
# Using the tensor vectors defined above, accessing the images in the dataset and passing them through the function defined above.

train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames_tensor, train_labels_tensor))
train_dataset = train_dataset.map(_parse_function)
# train_aug_dataset = train_aug_dataset.repeat(3)
train_dataset = train_dataset.batch(512)    # Same as batch_size hyperparameter in model.fit() below.

test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames_tensor, test_labels_tensor))
test_dataset = test_dataset.map(_parse_function)
# test_dataset = test_dataset.repeat(3)
test_dataset = test_dataset.batch(512)    # Same as batch_size hyperparameter in model.fit() below.

'''
# Defining the architecture of the sequential neural network.

final_cnn = Sequential()

# Input layer with 32 filters, followed by an MaxPooling2D layer.
final_cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(200, 200, 1)))    # 3rd dim = 1 for grayscale images.
final_cnn.add(MaxPooling2D(pool_size=(2,2)))

# Three Conv2D layers with filters increasing by a factor of 2 for every successive Conv2D layer.
final_cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
final_cnn.add(MaxPooling2D(pool_size=(2,2)))

final_cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
final_cnn.add(MaxPooling2D(pool_size=(2,2)))

final_cnn.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
final_cnn.add(MaxPooling2D(pool_size=(2,2)))

# A GlobalMaxPooling2D layer before going into Dense layers below.
# GlobalMaxPooling2D layer gives no. of outputs equal to no. of filters in last Conv2D layer above (256).
final_cnn.add(GlobalMaxPooling2D())

# One Dense layer with 132 nodes so as to taper down the no. of nodes from no. of outputs of GlobalMaxPooling2D layer above towards no. of nodes in output layer below (7).
final_cnn.add(Dense(132, activation='relu'))

# Output layer with 7 nodes (equal to the no. of classes).
final_cnn.add(Dense(7, activation='softmax'))

final_cnn.summary()


# Compiling the above created CNN architecture.

final_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fitting the above created CNN model.
'''

final_cnn = load_model('my_model_og2.h5')

final_cnn_history = final_cnn.fit(train_dataset,
                                  validation_data=test_dataset,
                                  epochs=2,
                                  shuffle=False    # shuffle=False to reduce randomness and increase reproducibility
                                 )


final_cnn.save('my_model_og3.h5') 
