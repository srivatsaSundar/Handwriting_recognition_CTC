import numpy as np
import cv2
import glob
# import simplejson
import os
import cv2
import csv
import sys
# import unidecode

from .helpers import implt
from .normalization import letter_normalization
from .viz import print_progress_bar

CHARS = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
         'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
         'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
         'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
         'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6',
         '7', '8', '9', '.', '-', '+', "'"]
CHAR_SIZE=len(CHARS)
idxs=[i for i in range(CHAR_SIZE)]
idx_2_chars=dict(zip(idxs,CHARS))
chars_2_idx=dict(zip(CHARS,idxs))

def char2idx(c,sequence=False):
    if sequence:
        return char2idx[c]+1
    return chars_2_idx[c]

def idx2char(i,sequence=False):
    if sequence:
        return idx_2_chars[i-1]
    return idx_2_chars[i]

def load_words_data(dataloc='data/words',is_csv=False,load_gaplines=False):
    print("Loading words")
    if type(dataloc) is not list:
        dataloc=[dataloc]

    if is_csv:
        csv.field_size_limit(sys.maxsize)
        length=0
        for loc in dataloc:
            with open(loc) as csvfile:
                reader=csv.reader(csvfile)
                length+=max(sum(1 for row in csvfile)-1,0)

        labels=np.empty(length,dtype=object)
        images=np.empty(length,dtype=object)
        i=0
        for loc in dataloc:
            with open(loc) as csvfile:
                reader=csv.reader(csvfile)
                for row in reader:
                    shape=np.fromstring(row['shape'],dtype=int,sep=' ')
                    img=np.fromstring(row['image'],dtype=np.uint8,sep=' ')
                    labels[i]=row['label']
                    images[i]=img

                    print_progress_bar(i,length)
                    i+=1

    else:
        img_list=[]
        tmp_labels=[]
        for loc in dataloc:
            for img_path in glob.glob(os.path.join(loc,'*.png')):
                img=cv2.imread(img_path)
                img_list.append(img)
                tmp_labels.append(os.path.basename(img_path).split('.')[0])

