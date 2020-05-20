#coding:utf-8
import re
import os
import pickle

import numpy as np
import cv2




class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

if __name__=='__main__':
    with open('onebyone_train.pickle', 'rb') as f:
        data = pickle.load(f)

    color_images = data['color_image']
    uv_images = data['uv_image']
    features = data['feature']
    numbers = data['number']
    #print(features)
    imgrgb = []
    rgbsum = np.zeros((3,3))
    for i in range(len(color_images)):
        img = color_images[i]
        feature = features[i]
        average_color_per_row = np.average(img, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        average_color = np.uint8(average_color)

        imgrgb.append(average_color)
        rgbsum[feature] += average_color
    #print(rgbsum)
    rgbmean = rgbsum / len(color_images) * 3
    #print(color_images[i])

    for i in range(len(color_images)):
        img = color_images[i]
        feature = features[i]
        sa = rgbmean[feature] - imgrgb[i]

        b,g,r = cv2.split(img)
        #if i==0:
        #    print(b)
        b = np.abs(b + sa[0])
        g = np.abs(g + sa[1])
        r = np.abs(r + sa[2])
        
        b = np.where(b<=255, b, 255)
        g = np.where(g<=255, g, 255)
        r = np.where(r<=255, r, 255)
        #if i==0:
        #    print(b)
        if not(np.all(b<=255) and np.all(g<=255) and np.all(r<=255)):
            print("OUT")
        color_images[i] = cv2.merge((b,g,r))
    #print(color_images[i])

    dataset = {
        'color_image' : np.array(color_images),
        'uv_image' : np.array(uv_images),
        'feature' : np.array(features),
        'number' : np.array(numbers),
    }
    
    
    filepath = 'onebyone_train_meanchange.pickle'
    pickle_dump(dataset, filepath)
    """
    with open('onebyone_train_meanchange.pickle', 'wb') as f:
        #4 is protcolversion
        pickle.dump(dataset, f, 4)
    """
