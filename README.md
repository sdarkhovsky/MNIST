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
- adapted the IPython script to Python.
- added command line parameters
- adopted drawing training and testing computational graphs to Python and added a few miscellaneous visualizations.
- added classification of a single image. The image should be an RGB image of an arbitrary size

The default number of training iterations total_iters is 200. My experiments show that it's not nearly enough 
to classify an arbitrary handwritten symbol, even though the final training statistics had 0.92 accuracy. 
See for example correctly predicted data/4.png extracted from training data and incorrectly predicted data/4a.png

