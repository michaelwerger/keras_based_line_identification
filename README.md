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
Matthieu Deru, Alassane Ndiaye: Deep Leaning with Tensorflow Keras und Tensorflow.js, Rheinwerk Verlag, 2020
Joachim Steinwendner, Roland Schwaiger, Neuronale Netze programmieren mit Python, Rheinwerk Verlag, 2020

Michael Werger, July 2023