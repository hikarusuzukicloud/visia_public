#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import random
import numpy as np
import pickle

import tensorflow as tf
import tensorflow.python.platform
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, Adadelta
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU
import keras.backend as K
from keras.utils import np_utils

import hyouka_visia as hyouka
import learge_pickle_dump as learge

NUM_CLASSES = 2
IMAGE_SIZE = 128
CELL_SIZE = 128
IMAGE_CHANNEL = 3
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNEL
BATCHSIZE = 32
EPOCHS = 200
#start = 1e-2
#start = 2.5*1e-3
start = 1*1e-4
stop = 1e-5
pickle_bool = True
load_weight = False


train_pickle = 'size128v2_train_ready_onebyone_suitmin_meanchange_rank2.pickle'
test_pickle = 'size128v2_test_ready_onebyone_suitmin_meanchange_rank2.pickle'

defsavepath = '183visia_main_200_a09b08c08f05_meanchange_adam_noskj_size{}_class{}'.format(IMAGE_SIZE, NUM_CLASSES)
weightsavefile = defsavepath + '.hdf5'
os.makedirs('./weight_result_main/'+defsavepath, exist_ok=True)
fileyouso = './weight_result_main/'+ defsavepath + '/' + defsavepath + '_'

weight_file = 'good_weight/visia_main_200_f05__512_size32to32_class3weight.50-0.60-0.68.hdf5'

os.makedirs('./plot/'+defsavepath, exist_ok=True)
graph_file = {
    'acc' : './plot/'+defsavepath+'/accuracy_' + defsavepath + '.png',
    'loss' : './plot/'+defsavepath+'/loss_' + defsavepath + '.png',
    'f1_score' : './plot/'+defsavepath+'/f1-score_' + defsavepath + '.png',
    'precision0' : './plot/'+defsavepath+'/precision0_' + defsavepath + '.png',
    'precision1' : './plot/'+defsavepath+'/precision1_' + defsavepath + '.png',
    'precision2' : './plot/'+defsavepath+'/precision2_' + defsavepath + '.png',
    'recall0' : './plot/'+defsavepath+'/recall0_' + defsavepath + '.png',
    'recall1' : './plot/'+defsavepath+'/recall1_' + defsavepath + '.png',
    'recall2' : './plot/'+defsavepath+'/recall2_' + defsavepath + '.png',
    'f1_0' : './plot/'+defsavepath+'/f1_0_' + defsavepath + '.png',
    'f1_1' : './plot/'+defsavepath+'/f1_1_' + defsavepath + '.png',
    'f1_2' : './plot/'+defsavepath+'/f1_2_' + defsavepath + '.png'
}

categories = ['acc', 'loss', 'f1_score',
              'precision0', 'recall0', 'f1_0',
              'precision1', 'recall1', 'f1_1',]
#              'precision2', 'recall2', 'f1_2']

input_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL)

def inputdata_pickle(im, lb):
    images = []
    labels = []
    for (image, label) in zip(im, lb):
        image.flatten().astype(np.float32)/255.0
        images.append(image)

        labels.append(label)

    images = np.asarray(images)
    labels = np.asarray(labels)
    labels = np_utils.to_categorical(labels, NUM_CLASSES)

    return images, labels
def makeplot(category, filename):
    val_category = 'val_'+category
    plt.plot(history.history[category])
    plt.plot(history.history[val_category])
    plt.title('model'+category)
    plt.ylabel(category)
    if category=='loss':
        plt.ylim([0, 0.7])
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(filename)
    plt.figure()

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.90))

model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.90))
"""
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.90))

model.add(Conv2D(filters=256, kernel_size=(2, 2), padding='same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
"""
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(NUM_CLASSES, activation = 'softmax'))

#データの入力
if pickle_bool:
    data = learge.pickle_load(train_pickle)
    train_im = data['train_image']
    train_lb = data['train_label']
    x_train, y_train = inputdata_pickle(train_im, train_lb)
    data = learge.pickle_load(test_pickle)
    test_im = data['test_image']
    test_lb = data['test_label']
    x_test, y_test = inputdata_pickle(test_im, test_lb)

if load_weight:
    model.load_weights(weight_file)

adam = Adam(lr=start)
#sgd=SGD(lr=start, momentum=0.9, nesterov=True)
model.compile(optimizer=adam,
            loss='binary_crossentropy',
            metrics=['accuracy', hyouka.precision, hyouka.recall, hyouka.f1_score,
            hyouka.precision0, hyouka.recall0, hyouka.f1_0,
            hyouka.precision1, hyouka.recall1, hyouka.f1_1,])
#            hyouka.precision2, hyouka.recall2, hyouka.f1_2])

learning_rates = np.linspace(start, stop, EPOCHS)
change_lr = LearningRateScheduler(lambda epoch:float(learning_rates[epoch]))

early_stop = EarlyStopping(patience=200)

checkpointfile = fileyouso + 'weight.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath = checkpointfile, save_best_only=False, mode='auto')

#学習の実行
"""
history = model.fit(x_train, y_train, batch_size=BATCHSIZE, epochs=EPOCHS,
          verbose=1, validation_data=(x_test, y_test), 
          callbacks=[change_lr, early_stop, checkpoint])
"""
history = model.fit(x_train, y_train, batch_size=BATCHSIZE, epochs=EPOCHS,
          verbose=1, validation_data=(x_test, y_test), 
          callbacks=[early_stop, checkpoint])

score = model.evaluate(x_test, y_test, verbose=0)

for c in categories:
    filename = graph_file[c]
    makeplot(c, filename)
