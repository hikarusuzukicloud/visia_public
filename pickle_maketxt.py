#coding:utf-8

import re
import os
import random
import pickle

import numpy as np
import cv2

import learge_pickle_dump as learge

NUM_CLASSES = 3

trainmax = 4863
testmax = 1146

load_train_filename = '128v2train_onebyone_suitmin_meanchange_rank3.pickle'
load_test_filename = '128v2test_onebyone_suitmin_meanchange_rank3.pickle'
ready_train_filename = 'size128v2_train_ready_onebyone_suitmin_meanchange_rank3.pickle'
ready_test_filename = 'size128v2_test_ready_onebyone_suitmin_meanchange_rank3.pickle'

def shuffle_two_list(x, y):
    zipped = list(zip(x, y))
    np.random.shuffle(zipped)
    nx, ny = zip(*zipped)
    return np.array(nx), np.array(ny)

def two_class(images, labels, maxt):
    train_images = []
    train_labels = []
    counttrain = [0 for i in range(4)]
    c=0
    for (image, label) in zip(images, labels):
        if label != 1:
            if counttrain[label] < maxt:
                if label == 2:
                    label = 1
                if c<10:
                    cv2.imwrite("asd{}.jpg".format(c), image)
                    print(label)
                    c+=1
                train_images.append(image)
                train_labels.append(label)
                counttrain[label] += 1
            elif min(counttrain) == maxt:
                break
    print(counttrain)
    return train_images, train_labels

def three_class(images, labels, maxt):
    train_images = []
    train_labels = []
    counttrain = [0 for i in range(4)]
    c=0
    for (image, label) in zip(images, labels):
        if counttrain[label] < maxt:
            if c<10:
                cv2.imwrite("asd{}.jpg".format(c), image)
                print(label)
                c+=1
            train_images.append(image)
            train_labels.append(label)
            counttrain[label] += 1
        elif min(counttrain) == maxt:
            break
    print(counttrain)
    return train_images, train_labels


if __name__=='__main__':
    #with open('train_onebyone_suitmin_meanchange_rank3.pickle', 'rb') as ftrain2:
    data = learge.pickle_load(load_train_filename)
    #data = pickle.load(ftrain2)

    images10 = data['train_image']
    labels10 = data['train_label']
    train_images = []
    train_labels = []

    images0, labels0 = shuffle_two_list(images10, labels10)

    if NUM_CLASSES==2:
        train_images, train_labels = two_class(images0, labels0, trainmax)

    elif NUM_CLASSES==3:
        train_images, train_labels = three_class(images0, labels0, trainmax)
    
    train_imageset = {
        'train_image' : np.array(train_images),
        'train_label' : np.array(train_labels),
    }

    learge.pickle_dump(train_imageset, ready_train_filename)
    """
    with open('train_ready_onebyone_suitmin_rank2.pickle', 'wb') as ftrain_save:
        pickle.dump(train_imageset, ftrain_save, 4)
    print(counttrain)
    """    
    #with open('test_onebyone_suitmin_meanchange_rank3.pickle', 'rb') as ftest2:
    data = learge.pickle_load(load_test_filename)
    #data = pickle.load(ftest2)

    images1 = data['test_image']
    labels1 = data['test_label']
    test_images = []
    test_labels = []

    images, labels = shuffle_two_list(images1, labels1)

    if NUM_CLASSES==2:
        test_images, test_labels = two_class(images1, labels1, testmax)

    elif NUM_CLASSES==3:
        test_images, test_labels = three_class(images1, labels1, testmax)


    test_imageset = {
        'test_image' : np.array(test_images),
        'test_label' : np.array(test_labels),
    }

    learge.pickle_dump(test_imageset, ready_test_filename)

    """
    with open('test_ready_onebyone_suitmin_rank2.pickle', 'wb') as ftest_save:
        pickle.dump(test_imageset, ftest_save, 4)
    print(counttest)
    """

