# MNIST
The repository is largely based on https://github.com/caffe2/tutorials/blob/master/MNIST.ipynb
See also https://caffe2.ai/docs/tutorial-MNIST.html.

Caffe2 has to be built with USE_LMDB defined to use the training and testing samples from an lmdb database.
By default Caffe2 is built without LMDB support. To build Caffe2 with LMDB support for example in Ubuntu 16.04 
follow  instructions from https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile, 
but execute the command    

export USE_LMDB=1   

before starting the build with    

python setup.py install --user

The following changes have been made to the original script:
- added command line parameters
- modified drawing the training and testing computational graphs and added a few miscellaneous visualizations.
- added classification of a real handwritten image (not from the MNIST database). The image should be an RGB image of 
  an arbitrary size. The image should contain a single digit. The program automatically recognizes whether the image 
  should be inverted during preprocessing. The algorithm for preprocessing was largely taken from  
  https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4 
  The preprocessing is necessary for successful recognition.
