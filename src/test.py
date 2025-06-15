# %%
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2

import shutil
import glob
import random



# %%
TRAIN_DATA_PATH= os.path.join(os.getcwd(), '..','data','train')
TEST_DATA_PATH = os.path.join(os.getcwd(), '..','data','test')
MODEL_PATH = os.path.join(os.getcwd(), '..','data','model.h5')

IMAGE_SIZE = 32; 


# %%
# Lädt ein Bild
def load_image(path,image_size=IMAGE_SIZE):
    img =cv2.imread(path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp = img.reshape([IMAGE_SIZE, IMAGE_SIZE,1])


    return np.array(tmp)/255

# %%
# Test auf Dataset
def test_dataset(model, labels_decoded, path, image_size=IMAGE_SIZE):
    correct_matches = 0
    result_table = PrettyTable()
    result_table.field_names = ["Datei ", "Ist", "Dekodiert", "Match?"]

    for filename in sorted(os.listdir(path)):

        if(filename.startswith('.') == False):

            current_wavelength = filename[0:7]
            print (current_wavelength)
            image_path = os.path.join(path,filename)
                
            test_image = load_image(image_path)
            predictions = model.predict(test_image.reshape((1,IMAGE_SIZE,IMAGE_SIZE,1)))
                
            index_max_predictions = np.argmax(predictions)
            print('index_max_predictions:',index_max_predictions, current_wavelength, labels_decoded[index_max_predictions])
            decode_wavelength = labels_decoded[index_max_predictions]

            # Passt oder nicht?
            if( str.upper(current_wavelength) == str.upper(decode_wavelength)):
                result_table.add_row([image_path, current_wavelength, decode_wavelength, "✅" ])
                correct_matches = correct_matches + 1 
            else:
                result_table.add_row([image_path, current_wavelength, decode_wavelength, "❌" ])


    print(result_table)

# %%
# copy some training data to directory test, if no test data is provided
#
# shutil.rmtree(TEST_DATA_PATH)
# os.mkdir(TEST_DATA_PATH)
# for directory in sorted(os.listdir(TRAIN_DATA_PATH)):
#     if(directory.startswith('.') == False):
#         p = os.path.join(TRAIN_DATA_PATH,directory)
#         for filename in sorted(os.listdir(p)):
#             if random.random() > 0.99:
#                
#                 shutil.copyfile(os.path.join(p,filename),os.path.join(TEST_DATA_PATH,filename))

        

        



# %%
labels_decoded = []
for directory in sorted(os.listdir(TRAIN_DATA_PATH)):
    if(directory.startswith('.') == False):
        labels_decoded.append(directory)

model = load_model(MODEL_PATH)
test_dataset(model, labels_decoded, TEST_DATA_PATH)

# %%



