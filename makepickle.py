#coding:utf-8
import re
import os
import pickle

import numpy as np
import cv2
from PIL import Image, ImageEnhance

IMAGE_SIZE = 32
save_pickle_file = 'oldvisia_train.pickle'

FACIAL_FEATURES = {
    0:u"forehead",
    1:u"rightcheak",
    2:u"leftcheak"
}
"""
dataset = {
    'color_image' : np.array(color_images),
    'uv_image' : np.array(uv_images),
    'feature' : np.array(features),
    'number' : numbers,
}
"""

color_images=[]
uv_images=[]
features=[]
numbers=[]

dir_numbers = []
for i in range(1, 24):
    if i < 10:
        n = '0{}'.format(i)
    else:
        n='{}'.format(i)
    dir_numbers.append(n)

for i in range(0, 23):
    imagedir = u'/Volumes/ボリューム/oldvisia/train_NOV6-{}/'.format(dir_numbers[i])
    images = [image for image in os.listdir(imagedir)]
    #画像がどこの部分の画像かを仕分ける
    color = [0 for i in range(3)]
    uv = [0 for i in range(3)]

    for imagepath in images:
        for number, name in FACIAL_FEATURES.items():
            if re.search(name, imagepath):
                if re.search(u'color', imagepath):
                    color[number] = imagepath
                    break
                elif re.search(u'uv', imagepath):
                    uv[number] = imagepath
                    break

    for k in range(3):
        color_len = color[k].split("_")
        uv_len = uv[k].split("_")
        if len(color_len)==5:
            color[k]="_".join(color_len[1:5])
        if len(uv_len)==5:
            uv[k]="_".join(uv_len[1:5])
    
    for k in range(3):
        color_imagesrc = os.path.join(imagedir, color[k])
        color_image = cv2.imread(color_imagesrc, 1)

        uv_imagesrc = os.path.join(imagedir, uv[k])
        uv_image = Image.open(uv_imagesrc)
        uv_image_gray = uv_image.convert("L")
        contrast_converter = ImageEnhance.Contrast(uv_image_gray)
        uv_image_con = contrast_converter.enhance(1.6)
        uv_image_con_tocv = np.asarray(uv_image_con)
        #uv_image_con_tocv = uv_image_con_tocv[:, :, ::-1].copy()

        color_images.append(color_image)
        uv_images.append(uv_image_con_tocv)
        features.append(k)
        numbers.append(i+1)

dataset = {
    'color_image' : np.array(color_images),
    'uv_image' : np.array(uv_images),
    'feature' : np.array(features),
    'number' : np.array(numbers),
}

with open(save_pickle_file, 'wb') as f:
    #4 is protcolversion
    pickle.dump(dataset, f, 4)
