# %%
%load_ext tensorboard

#
#
# start Tensorboard with CMD-Shft-P Python:Launch TensorBoard

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

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.utils import shuffle

import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(),'..'))
from lib import find_nearest_index, FigureSize




# %%
logs_base_dir = "./logs"
os.makedirs(logs_base_dir, exist_ok=True)

# %%
NUM_OUTPUTS = 26 # no. of peaks
NUM_BATCHES = 32
NUM_EPOCHS = 100
IMAGE_SIZE = 32
#NUM_TRAIN_LABELS = 2600 # use outpuf of load_images()

TRAIN_DATA_PATH= os.path.join(os.getcwd(), '..','data','train')
TEST_DATA_PATH = os.path.join(os.getcwd(), '..','data','test')
MODEL_PATH = os.path.join(os.getcwd(), '..','data','model.h5')

TESTDATA_TRAINDATA_RATIO = 20./80.

# %%
os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
os.makedirs(TEST_DATA_PATH, exist_ok=True)

# %%
NEON_REFERENCE_FILE = os.path.join(os.getcwd(),'..','data','ref','NIST','Ne','neon-exported.csv')
WAVELENGTHS_MIN, WAVELENGTHS_MAX = 4000, 9000
WINDOW = 512
STEPSIZE_MIN, STEPSIZE_MAX, STEPSIZE_N = 0.5, 1.5, 2000
INTENSITY_SCALE = 255

# %%
RANDOM_NUMBER = math.pow(10,int(math.log10(STEPSIZE_N *NUM_OUTPUTS))+1)
print (RANDOM_NUMBER)

# %%
def scale(min_scale, max_scale):
    return random.random()*(max_scale - min_scale)+0.95

# %%
def gauss():
    return random.gauss(1.0,0.3)

# %%
def build_model():
        
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((IMAGE_SIZE,IMAGE_SIZE,1),input_shape=(IMAGE_SIZE,IMAGE_SIZE,1)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))         
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(NUM_TRAIN_LABELS, activation='softmax')) 
    model.add(tf.keras.layers.Dense(NUM_OUTPUTS, activation='softmax')) 

    return model

# %%
def load_images(path):
    train_images = []
    train_labels = []

    dir_index = 0
    for directory in sorted(os.listdir(path)):
        if(directory.startswith('.') == False):

            #print("Buchstabe: {}".format(LIST_OF_CHARS[dir_index]))
            for filename in sorted(os.listdir(os.path.join(path, directory))):

                if(filename.startswith('.') == False):
                        
                        image_path = os.path.join(path, directory,filename    )
                        img =cv2.imread(image_path)
                        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = img.reshape([IMAGE_SIZE, IMAGE_SIZE,1])
                        train_images.append(img)
                        train_labels.append(dir_index)
            dir_index = dir_index + 1

    print(len(train_images),train_labels)
    return np.array(train_images)/255, np.array(tf.keras.utils.to_categorical(train_labels,NUM_OUTPUTS))#len(train_labels)))


# %%
neon_reference_file = os.path.join(NEON_REFERENCE_FILE)
positions = {'wavelength':2, 'intensity':6, 'selector':1}
selector = '1'
intensity_limit = 1.0
neon_wavelengths = []
neon_intensities = []

with open(neon_reference_file,'r') as neon_f:
    for line in neon_f:
        if line.startswith('#'):
            pass
        else:
            tokens =  line.split(';')
            #print (tokens)
            try:
                if selector in tokens[positions['selector']] :
                    
                    neon_wavelength, neon_intensity = float(tokens[positions['wavelength']]), float(tokens[positions['intensity']])
                    
                    if neon_intensity > intensity_limit :
                        
                        neon_wavelengths.append(neon_wavelength)
                        neon_intensities.append(neon_intensity)
                        
                        print ("{:8.3f} {:10.0f}".format(neon_wavelength, neon_intensity, ))
                    
            except ValueError:
                pass

print (len(neon_wavelengths), len(neon_intensities))
#

# %%
neon_reference_file = 'linetable-NE.csv'
positions = {'wavelength':0, 'intensity':2}
neon_wavelengths = []
neon_intensities = []

with open(neon_reference_file,'r') as neon_f:
    for line in neon_f:
        if line.startswith('#'):
            pass
        else:
            tokens =  line.split(';')
            print (tokens)
            try:
                neon_wavelength, neon_intensity = float(tokens[positions['wavelength']]), float(tokens[positions['intensity']])
                neon_wavelengths.append(neon_wavelength)
                neon_intensities.append(neon_intensity)
                
                print ("{:8.3f} {:10.0f}".format(neon_wavelength, neon_intensity, ))
                
                        
            except ValueError:
                pass

print (len(neon_wavelengths), len(neon_intensities))
#
# print (neon_wavelengths)

# %%
plt.rcParams['figure.figsize'] = FigureSize.NARROW

# %%
wavelengths = np.array(range(WAVELENGTHS_MIN,WAVELENGTHS_MAX,1))*1.0
intensities = wavelengths * 0.0
sigma = 4.0
k = -2*sigma*sigma
r = math.sqrt(2*math.pi*sigma*sigma)

for neon_w, neon_i, in zip(neon_wavelengths, neon_intensities):

    _e = (wavelengths - neon_w)*(wavelengths- neon_w) / k
    intensities = intensities + np.exp(_e) * neon_i
        

xlim = [wavelengths[1], wavelengths[-2]]
_i1 = find_nearest_index(wavelengths,xlim[0])
_i2 = find_nearest_index(wavelengths,xlim[1])
#print (_i1[0], _i2[0])
max_i = intensities[_i1:_i2].max()
normalized_intensities = intensities / max_i
fig, ax = plt.subplots()

plt.plot(wavelengths, normalized_intensities)
for _nw, _ni in zip(neon_wavelengths, neon_intensities):
    plt.text(_nw,_ni/max_i,"{:6.1f}".format(_nw), rotation=90, horizontalalignment='center')
plt.xlim(xlim)
plt.ylim(0,1.1)
plt.show()

# %%
wavelengths = np.array(range(WAVELENGTHS_MIN,WAVELENGTHS_MAX,1))*1.0
pixels = np.array(range(0,len(wavelengths)))
intensities = wavelengths * 0.0
sigma = 4.0
k = -2*sigma*sigma
r = math.sqrt(2*math.pi*sigma*sigma)

for neon_w, neon_i, in zip(neon_wavelengths, neon_intensities):

    _e = (wavelengths - neon_w)*(wavelengths- neon_w) / k
    intensities = intensities + np.exp(_e) * neon_i

xlim = [wavelengths[1], wavelengths[-2]]
_i1 = find_nearest_index(wavelengths,xlim[0])
_i2 = find_nearest_index(wavelengths,xlim[1])
max_i = intensities[_i1:_i2].max()
normalized_intensities = intensities / max_i

ny = WINDOW
nx = len(normalized_intensities)

twod = np.zeros((ny, nx))
for i in range(ny):
    twod[i] = normalized_intensities*-1+1.0

ylim = [0,ny]

n_cols = 5
n_rows = math.ceil(len(neon_wavelengths)/n_cols)

window = WINDOW
window_h = int(window/2)
image_size = [IMAGE_SIZE,IMAGE_SIZE]

plt.rcParams['figure.figsize'] = FigureSize.LARGE
fig, axes = plt.subplots(n_rows, n_cols,sharex=False, sharey=False)

i_row = 0
i_col = 0

for index in range(0,len(neon_wavelengths)):
    neon_w = neon_wavelengths[index]
    
    _xlim = [
            find_nearest_index(wavelengths,neon_w)-window_h,
            find_nearest_index(wavelengths,neon_w)+window_h,
    ]
    _ylim = [0,ny]
    _twod = twod[_ylim[0]:_ylim[1], _xlim[0]:_xlim[1]]
    wavelength_text = str(int(neon_w*100)/100)
    print(index, neon_w, wavelength_text, _xlim)
    _res = cv2.resize(np.uint8(_twod * INTENSITY_SCALE), dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    if n_cols == 1:
        axes[i_row].imshow(_res, cmap='gray')
        axes[i_row].set_title(wavelength_text)
        #axes[i_row].set_ylim(_ylim)
        #axes[i_row].set_xlim(0,_xlim[1]-_xlim[0])
        print (i_row, wavelength_text, _xlim)
        i_row += 1
    else:
        axes[i_row, i_col].imshow(_res, cmap='gray')
        axes[i_row, i_col].set_title(wavelength_text)
        #axes[i_row, i_col].set_ylim(_ylim)
        #axes[i_row, i_col].set_xlim(0,_xlim[1]-_xlim[0])

        i_col += 1
        if i_col >= n_cols:
            i_col = 0
            i_row +=1
plt.show()

# %%
for neon_w in neon_wavelengths:
    wavelength_text = str(int(neon_w*100)/100)
    p = os.path.join(TRAIN_DATA_PATH, wavelength_text)
    try:
        shutil.rmtree(p, ignore_errors=True)
        
    except FileNotFoundError:
        print ("skipping "+p)
    os.makedirs(p) # ensure creation of parent dirs

# %%
sigma = 4.0
k = -2*sigma*sigma
r = math.sqrt(2*math.pi*sigma*sigma)

xlim = [wavelengths[1], wavelengths[-2]]
ny = WINDOW
nx = len(normalized_intensities)
ylim = [0,ny]

window_h = int(WINDOW/2)
image_size = [IMAGE_SIZE,IMAGE_SIZE]

filename_counter = [0] * len (neon_wavelengths)

min_stepsize = STEPSIZE_MIN
max_stepsize = STEPSIZE_MAX
n_steps = STEPSIZE_N
d_step = (max_stepsize - min_stepsize)/n_steps

stepsizes = np.array(range(n_steps))*d_step+min_stepsize
#print (stepsizes)
nbins = WAVELENGTHS_MAX - WAVELENGTHS_MIN
for stepsize in stepsizes:
    wavelengths = np.array(range(nbins))*stepsize+WAVELENGTHS_MIN #   np.array(range(4000,9000,stepsize))*1.0
    
    intensities = np.zeros(nbins)

    for neon_w, neon_i, in zip(neon_wavelengths, neon_intensities):
        _neon_w = neon_w+random.gauss(0,1.5)
        _e = (wavelengths - _neon_w)*(wavelengths- _neon_w) / k
        intensities = intensities + np.exp(_e) * neon_i*gauss() # scale(0.9,1.0)

    max_i = intensities[1:-2].max()
    normalized_intensities = intensities / max_i

    twod = np.zeros((ny, nx))
    for i in range(ny):
        twod[i] = normalized_intensities*-1+1.0


    for index in range(0,len(neon_wavelengths)):
        neon_w = neon_wavelengths[index]
        
        _xlim = [
            find_nearest_index(wavelengths,neon_w)-window_h,
            find_nearest_index(wavelengths,neon_w)+window_h,
        ]
        _ylim = [0,ny]
        _twod = twod[_ylim[0]:_ylim[1], _xlim[0]:_xlim[1]]
        wavelength_text = str(int(neon_w*100)/100)
        
        res = cv2.resize(np.uint8(_twod * INTENSITY_SCALE), dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(res)

        prob = random.random()
        
        if prob < TESTDATA_TRAINDATA_RATIO:
            p = os.path.join(TEST_DATA_PATH,'{:s}.{:06d}.BMP'.format(wavelength_text,filename_counter[index]))
        else:
            p = os.path.join(TRAIN_DATA_PATH,wavelength_text,'{:s}.{:06d}.BMP'.format(wavelength_text,filename_counter[index]))

        img.save(p, format='BMP')
        
        filename_counter[index] += 1


    

# %%


# %%
train_images, train_labels  = load_images(TRAIN_DATA_PATH)


train_images, train_labels  = shuffle(train_images,train_labels,random_state=21)
print (len(train_labels))
xTrain, xTest, yTrain, yTest = train_test_split(train_images, train_labels, test_size = 0.2, random_state = 0)
optimizer = tf.keras.optimizers.Adam()

model = build_model()
model.summary()
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics = ["accuracy"])
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="logs",histogram_freq=5)
model.fit(xTrain,yTrain, epochs=NUM_EPOCHS,batch_size=NUM_BATCHES, callbacks=[tf_callback], verbose=1,validation_split=0.2)

results = model.evaluate(xTest,yTest,verbose=1)
print("--- Ergebnisse {} ----".format(TRAIN_DATA_PATH))
print('Evaluation / Loss {}, Acc:{}'.format(results[0],results[1]))
export_path = MODEL_PATH

model.save(export_path)
print("💾 Modell gespeichert")

# %%



