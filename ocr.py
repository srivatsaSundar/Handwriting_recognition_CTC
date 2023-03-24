import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from textblob import TextBlob
sys.path.append('../src')
from ocr.normalization import word_normalization, letter_normalization
from ocr import word,page,characters
from ocr.helpers import implt, resize
from ocr.tfhelpers import Model
from ocr.datahelpers import idx2char

# %matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 10.0)

#LANG = 'en'
# You can use only one of these two
# You HABE TO train the CTC model by yourself using word_classifier_CTC.ipynb
MODEL_LOC_CHARS = f'/home/srivatsa/Documents/Handwriting_recognition_CTC/ocr-handwriting-models/char-clas/en/CharClassifier'
MODEL_LOC_CTC = '/home/srivatsa/Documents/Handwriting_recognition_CTC/ocr-handwriting-models/word-clas/CTC/Classifier1'
print("Loading models...")

CHARACTER_MODEL = Model(MODEL_LOC_CHARS)
CTC_MODEL = Model(MODEL_LOC_CTC, 'word_prediction')
print("Models loaded.")

IMG="/home/srivatsa/Documents/Handwriting_recognition_CTC/media/IMG20230308114842.jpg"
img=cv2.imread(IMG)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# implt(image)

crop = page.detection(image)
# implt(crop)
boxes = word.detection(crop)
lines = word.sort_words(boxes)

def recognise(img):
    """Recognising words using CTC Model."""
    img = word_normalization(
        img,
        64,
        border=False,
        tilt=False,
        hyst_norm=False)
    length = img.shape[1]
    # Input has shape [batch_size, height, width, 1]
    input_imgs = np.zeros(
            (1, 64, length, 1), dtype=np.uint8)
    input_imgs[0][:, :length, 0] = img

    pred = CTC_MODEL.eval_feed({
        'inputs:0': input_imgs,
        'inputs_length:0': [length],
        'keep_prob:0': 1})[0]

    word = ''
    for i in pred:
        word += idx2char(i + 1)
    return word

#implt(crop)

print("Recognising text...")
print()
output=[]
for line in lines:
    outputs=" ".join([recognise(crop[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line])
    output.append(outputs)
    print(outputs)
    
print(output)    

# Reconstrucing the words to original form

for output in output:
    blob = TextBlob(output)
    print(blob.correct())