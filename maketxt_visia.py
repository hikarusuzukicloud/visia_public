#coding:utf-8

import re
import os
import random

import numpy as np
import cv2

counttrain = [0 for i in range(4)]
label_train_max = [4500, 6500, 4500, 1100]

with open('size8_rank4.txt', mode='r', encodeint='utf-8') as fall:
    with open('train_size8_rank4.txt', mode='w', encoding='utf-8') as ftrain:
        with open('test_size8_rank4.txt', mode='w', encoding='utf-8') as ftest:
            alldata=fall.readlines()
            random.shuffle(alldata)

            for data in alldata:
                data = data.split()
                image_path = data[0]
                label = data[1]

                if counttrain[label] < label_train_max[label]:
                    ftrain.write('%s %d\n' % (str(image_path), int(label)))
                else:
                    ftest.write('%s %d\n' % (str(image_path), int(label)))

                counttrain[label] += 1

print(counttrain)
