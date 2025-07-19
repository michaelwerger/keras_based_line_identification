# %%
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage, NonUniformImage
import numpy as np
import math
from scipy.interpolate import interp1d
from PIL import Image
import cv2
import shutil
import random
import uuid
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.utils import shuffle

import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(),'..'))
from lib import find_nearest_index, FigureSize

# %%
def getrowfromimage(fp):
    # img = cv2.imread(fp)
    # source_img = cv2.cvtColor(img,cvs.IMREAD_GRAYSCALE)
    # source_vec = source_img[0,:]

    with Image.open(fp) as img:
        source_img = np.array(img)
        #print(source_img.shape)
        _vec = source_img[0,:]
        #min = np.min(_vec)
        max = np.max(_vec)
        source_vec = _vec*(-1)+max

    return source_vec

# %%
def getcosphi(a_vec, b_vec):

    a_norm = np.linalg.norm(a_vec)
    b_norm = np.linalg.norm(b_vec)
            
    return (np.dot(a_vec,b_vec)/a_norm/b_norm)

# %%
def getrms(a_vec, b_vec):

    d = (a_vec - b_vec)

    return np.sqrt(np.dot(d,d))


# %%
source_file = '/Users/Micha_1/Workspaces/keras_based_line_identification/data/train/5852.48/5852.48.001986.BMP'
v = getrowfromimage(source_file)
plt.plot(v)

# %%
for dx in range(v.shape[0]):
    v_shifted = tf.Variable(np.roll(v,dx), tf.float16)
    cosphi = getcosphi(v,v_shifted)
    rms = getrms(v, v_shifted)
    print (f'{dx:4d} {cosphi:8.3f} {rms:8.3f} ')

# %%
import pickle



# %%
IMAGE_SIZE = 64

# %%
TRAIN_DATA_PATH= Path('/Users/Micha_1/Workspaces/keras_based_line_identification/data/train')
os.chdir(str(TRAIN_DATA_PATH))
current_path = Path('.')

files = sorted([_f for _f in current_path.glob('**/*.BMP')], reverse=True)
n = len(files)

data_path = TRAIN_DATA_PATH / 'array.pickle'
if data_path.exists():
    train_array = pickle.load(str(data_path))
else:
    train_array = np.ndarray((n,IMAGE_SIZE))
    print (train_array.shape)
    for i in range(n):
        train_array[i,:] = getrowfromimage(str(files[i]))



# %%

indexes = [x for x in range(n)]
indexes_copy = indexes.copy()

result = np.zeros((n,n,2), np.float16())
while len(indexes) > 0:
    f1 = indexes.pop()
    if f1 % 100 == 0:
        print (f1)
    for f2 in indexes:

        result[f1,f2,0] = getcosphi(train_array[f1], train_array[f2])
        result[f1,f2,1] = getrms(train_array[f1], train_array[f2])


with open('result.pickle','wb') as r_out:
    pickle.dump(result, r_out)

# %%



