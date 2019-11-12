# from __future__ import print_function

import tensorflow
from pyspark import SparkContext, SparkConf
from keras.utils.data_utils import get_file
import numpy as np
import os
import keras
from keras.datasets.cifar import load_batch
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
import numpy

conf = SparkConf().setAppName('elephas_app').setMaster('spark://hp:7077')
sc = SparkContext(conf=conf)

path = './cifar-10-batches-py/'

num_train_samples = 50000

x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
y_train = np.empty((num_train_samples,), dtype='uint8')

for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000: i * 10000, :, :, :],
     y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

fpath = os.path.join(path, 'test_batch')
x_test, y_test = load_batch(fpath)

y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)


batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# Let's train the model
model.compile(loss='categorical_crossentropy',
              optimizer=SGD())

rdd = to_simple_rdd(sc, x_train, y_train)

spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
spark_model.fit(rdd, epochs=2, batch_size=32, verbose=1, validation_split=0.1)