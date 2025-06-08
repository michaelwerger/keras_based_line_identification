# Overview

A Keras based line detection for automatic wavelength calibration

(this documentation will be more detailed.)

The idea behind this project is to automate line identification during 
calibration of spectroscopic data.
There may be already many available tools to do this. However, this 
project intends to combine the following aspects
* automate line identification for DADOS(TM) spectra
* apply machine learning like KERAS
* implement calibration on a mac with M1

The idea behind the line identification is simple. Like the human
eye which identifies lines by their line profile and their 
neighbouring lines left and right to the line, this algorithm 
applies feature detection of KERAS in a small convolution neural
network with a few layers.
To enable feature detection and learning, a synthetic Neon spectrum
is computed (using line data from NIST for the various transitions)
and split in serveral pieces: for each line in the spectum a part 
of the spectrum is extraced with a certain window width. This spectrum
part is used to create a 2-dim image repeating each part several times
having a 2-dim image with the same size in x- as in y-direction.
This images are then rebinned to a 32x32 size.
But how can we enable training of the convolutional neural network?
For each line, images are created where the wavelength scale is sligthly
stretched or shrunk. Again, images are created from this spectra with
different wavelenth scale. For each Neon line being the central line 
in the truncated spectrum such images are stored in a directory named 
according the central line:
* 5852.48: 5852.48.000000.BMP, 5852.48.000001.BMP, 5852.48.000002.BMP, ...
* 5881.89: 5881.89.000000.BMP, 5881.89.000001.BMP, 5881.89.000002.BMP, ...
* ...
each images in a directory belongs to the same central wavelength but each
image is stretched or shrunk in wavelength (x-) direction.

Next, training is used with a KERAS model using all created images. 
Labelling is done using the directory names of the files.

The model is saved as model.h5

To test the model, some of the training data is copied into a test
directory and the model is applied onto this test data. The 
evaluation is written using PrettyTable.

This version shows the implementation using synthetic data only. Using
real data will come next.

This project uses a python virtual environment to install and run 
tensorflow on Apple Mac M1

This project uses
* Jupyter Notebook
* Microsoft Visual Code
* data from NIST (NIST.gov)
* Tensorflow KERAS (tensorflow.org)
which are greatly appreciated!
  
The principal idea of the procedure is based on parts from the following 
books:
* Matthieu Deru, Alassane Ndiaye: Deep Leaning with Tensorflow Keras und Tensorflow.js, Rheinwerk Verlag, 2020
* Joachim Steinwendner, Roland Schwaiger, Neuronale Netze programmieren mit Python, Rheinwerk Verlag, 2020

# Creating the Python environment

It was a bit tricky to create an Python virtual environment which supports GPU usage on a Apple M1 pro.
Note: Tensorflow does not officially support Apple silicon GPUs in the recent versions. However, there is a
a way using a set of compatible libraries, python 3.11 and some little "tricks".

There are a lot of similar descriptions to enable GPU usage with tensorflow, but neither of them worked - they ended in kernel crashes, 
missing symbol problems, incompatibility issues with numpy and other libraries and so on, as documented here
* https://numpy.org/doc/stable/user/troubleshooting-importerror.html
* https://github.com/jax-ml/jax/discussions/19343
* https://github.com/jax-ml/jax/discussions/22289

The following worked for me for this project:

I use a directory ~/conda/py311 for the miniconda environment and 
a separate directory ~/conda/channel/apple for packages downloaded here

> cd ~
> cd conda

Download the installation script for minicona for Apple silicon from https://repo.anaconda.com/miniconda/ into ~/conda/py311,
then continue as follows

> bash ./py311/Miniconda3-py311_25.3.1-1-MacOSX-arm64.sh -b -u -p ~/conda/py311
> source py311/bin/activate

Then, switch to the project directory
> cd ~/Workspaces
> cd keras_based_line_identification
> conda create -n tf python=3.11.11
> conda activate tf

Downlaod tensorflow-deps-2.10.0-0.tar.bz2 from https://anaconda.org/apple/tensorflow-deps/files
> conda install ~/conda/channel/apple/tensorflow-deps-2.10.0-0.tar.bz2
> pip install tensorflow-metal
> pip install pandas
> pip install matplotlib
> pip install scikit-learn
> pip install scipy
> pip install 'imageio==2.37.0'
> pip install plotly
> pip install opencv-python
> conda install notebook

You may then use src/check_gpu.ipynb for an output similar to this:

```
Python Platform: macOS-15.5-arm64-arm-64bit
Tensor Flow Version: 2.16.2
Keras Version: 3.10.0

Python 3.11.11 (main, Dec 11 2024, 10:25:04) [Clang 14.0.6 ]
Pandas 2.3.0
Scikit-Learn 1.7.0
SciPy 1.15.3
GPU is available
```

Michael Werger, June 2025