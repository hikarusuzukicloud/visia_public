#coding:utf-8
import re
import os
import pickle

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps

import learge_pickle_dump as learge

IMAGE_SIZE = 32
idou = int(IMAGE_SIZE/2)
distort_bool = False
check_label = False

FACIAL_FEATURES = {
    0:u"forehead",
    1:u"right",
    2:u"left"
}

def label_count(image):
    count = [0 for i in range(4)]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gaso = image[i][j]
            if gaso < 60:
                count[0] += 1
            elif gaso < 90:
                count[1] += 1
            elif gaso <110:
                count[2] += 1
            else:
                count[3] += 1

    if count[0] > 384:
        label = 0
    elif (count[0]+count[1]) > 512:
        label = 1
    elif (count[0] < 128) and (count[0]+count[1] < 256):
        label = 2
    else:
        label = 10
    
    return label


def create_distort_image(img):
    distorts = []

    contrast_converter = ImageEnhance.Contrast(img)
    for i in range(3):
        img_distort = contrast_converter.enhance(1+0.4*i)
        distorts.append(img_distort)
        img_distort = ImageOps.mirror(img_distort)
        distorts.append(img_distort)

    saturation_converter = ImageEnhance.Color(img)
    img_distort = saturation_converter.enhance(1.4)
    distorts.append(img_distort)
    img_distort = ImageOps.mirror(img_distort)
    distorts.append(img_distort)

    return distorts

def create_label_check_image(img, i, j, label):
    # 明度を変える
    if label == 0:
        gaso = (255, 0, 0)
        cv2.rectangle(img, (j*idou, i*idou), (j*idou + IMAGE_SIZE, i*idou + IMAGE_SIZE), gaso, -1)
    elif label == 1:
        gaso = (0, 255, 0)
        cv2.rectangle(img, (j*idou, i*idou), (j*idou + IMAGE_SIZE, i*idou + IMAGE_SIZE), gaso, -1)
    elif label == 2:
        gaso = (0, 0, 255)
        cv2.rectangle(img, (j*idou, i*idou), (j*idou + IMAGE_SIZE, i*idou + IMAGE_SIZE), gaso, -1)
    return img

def create_label(face, simi, dir_number, image_list, label_list, counttrain):
    tmp_image_list = [ [] for i in range(3)]
    tmp_label_list = [ [] for i in range(3)]
    count_dir = [0 for i in range(3)]
    for k in range(3):
        face_image = face[k]
        simi_image = simi[k]

        if dir_number == 1:
            cv2.imwrite('hitai.jpg', simi[0])

        #画像の縦と横の長さから切りとれる枚数を測る
        col = int(simi_image.shape[0]*2/IMAGE_SIZE) -1
        row = int(simi_image.shape[1]*2/IMAGE_SIZE) -1

        for i in range(col):
            for j in range(row):
                #crop
                simi_cell = simi_image[i*idou:(i*idou + IMAGE_SIZE), j*idou:(j*idou + IMAGE_SIZE)]

                label = label_count(simi_cell)

                if check_label:
                    check_image = create_label_check_image(check_image, i, j, label)

                if label <10:
                    face_cell = face_image[i*idou:(i*idou + IMAGE_SIZE), j*idou:(j*idou + IMAGE_SIZE)]
                    count_dir[label] += 1
                    tmp_image_list[label].append(face_cell)
                else:
                    counttrain[label]+=1

    print(dir_number)
    print(count_dir)
    a = count_dir
    a.sort()
    maxc = min([a[0], 400])
    for i in range(3):
        np.random.shuffle(tmp_image_list[i])
        c = 0
        for image in tmp_image_list[i]:
            if c < maxc:
                image_list.append(image)
                label_list.append(i)
                c+=1
            else:
                break

    counttrain[0]+=maxc
    counttrain[1]+=maxc
    counttrain[2]+=maxc
    return image_list, label_list, counttrain

if __name__=='__main__':
    with open('onebyone_train_meanchange.pickle', 'rb') as f:
        data = pickle.load(f)

    color_images = data['color_image']
    uv_images = data['uv_image']
    features = data['feature']
    numbers = data['number']

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    traincount = [0 for i in range(20)]
    testcount = [0 for i in range(20)]

    for i in range(int(len(color_images)/3)):
        color = [color_images[i*3+0], color_images[i*3+1], color_images[i*3+2]]
        uv = [uv_images[i*3+0], uv_images[i*3+1], uv_images[i*3+2]]

        if i < 150:
            train_images, train_labels, traincount = create_label(color, uv, i+1, train_images, train_labels, traincount)
        else:
            test_images, test_labels, testcount = create_label(color, uv, i+1, test_images, test_labels, testcount)


    train_imageset = {
        'train_image' : np.array(train_images),
        'train_label' : np.array(train_labels),
    }
    test_imageset = {
        'test_image' : np.array(test_images),
        'test_label' : np.array(test_labels),
    }
    """
    with open('train_onebyone_suitmin_meanchange_rank3.pickle', 'wb') as ftrain:
        with open('test_onebyone_suitmin_meanchange_rank3.pickle', 'wb') as ftest:
            pickle.dump(train_imageset, ftrain, 4)
            pickle.dump(test_imageset, ftest, 4)
    """
    learge.pickle_dump(train_imageset, 'v2train_onebyone_suitmin_meanchange_rank3.pickle')
    learge.pickle_dump(test_imageset, 'v2test_onebyone_suitmin_meanchange_rank3.pickle')

    print(traincount)
    print(testcount)
