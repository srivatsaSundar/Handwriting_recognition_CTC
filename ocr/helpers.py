import matplotlib.pyplot as plt
import numpy as np
import cv2

# Path: ocr/helpers.py

SMALL_HEIGTH=800
def implt(img, cmp=None, t=''):
    plt.imshow(img, cmap=cmp)
    plt.title(t)
    plt.show()

def resize(img, height=SMALL_HEIGTH, allways=False):
    if allways or img.shape[0] > height:
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    return img

def ratio(img, height=SMALL_HEIGTH):
    return img.shape[0] / height

def immg(img,shape):
    x=np.zeros(shape,np.unit8)
    x[:img.shape[0],:img.shape[1]]=img
    return x