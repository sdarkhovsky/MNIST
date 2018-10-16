from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from matplotlib import pyplot
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import os
import shutil
import operator
import caffe2.python.predictor.predictor_exporter as pe
import argparse
import sys

from caffe2.python import (
    brew,
    core,
    model_helper,
    net_drawer,
    optimizer,
    visualize,
    workspace,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MNIST Tutorial'
    )
    parser.add_argument(
        '--train',
        dest='train',
        help='Train MNIST',
        action='store_true'
    )
    parser.add_argument(
        '--test',
        dest='test',
        help='Test MNIST',
        action='store_true'
    )
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='Image path',
        default=None,
        type=str
    )
    parser.add_argument(
        '--total_iters',
        dest='total_iters',
        help='total_iters',
        default=200,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

args = parse_args()
if (args.test and args.train) or (not args.test and not args.train):
   print("Test or train?")

# If you would like to see some really detailed initializations,
# you can change --caffe2_log_level=0 to --caffe2_log_level=-1
core.GlobalInit(['caffe2', '--caffe2_log_level=1'])
print("Necessities imported!")

# If True, use the LeNet CNN model
# If False, a multilayer perceptron model is used
USE_LENET_MODEL = True

# This section preps your image and test set in a lmdb database
def DownloadResource(url, path):
    '''Downloads resources from s3 by url and unzips them to the provided path'''
    import requests, zipfile, io
    print("Downloading... {} to {}".format(url, path))
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)
    print("Completed download and extraction.")
    
# Setup the paths for the necessary directories 
current_folder = os.path.join(os.path.expanduser('~'), 'mnist')
data_folder = os.path.join(current_folder, 'data')
workspace_folder = os.path.join(current_folder, 'workspace')
figure_folder = os.path.join(current_folder, 'figures')
db_missing = False

# Check if the data folder already exists
if not os.path.exists(data_folder):
    os.makedirs(data_folder)   
    print("Your data folder was not found!! This was generated: {}".format(data_folder))

if not os.path.exists(figure_folder):
   os.makedirs(figure_folder)

# Check if the training lmdb exists in the data folder
if os.path.exists(os.path.join(data_folder,"mnist-train-nchw-lmdb")):
    print("lmdb train db found!")
else:
    db_missing = True
    
# Check if the testing lmdb exists in the data folder   
if os.path.exists(os.path.join(data_folder,"mnist-test-nchw-lmdb")):
    print("lmdb test db found!")
else:
    db_missing = True

# Attempt the download of the db if either was missing
if db_missing:
    print("one or both of the MNIST lmbd dbs not found!!")
    db_url = "http://download.caffe2.ai/databases/mnist-lmdb.zip"
    try:
        DownloadResource(db_url, data_folder)
    except Exception as ex:
        print("Failed to download dataset. Please download it manually from {}".format(db_url))
        print("Unzip it and place the two database folders here: {}".format(data_folder))
        raise ex

if args.train:
	# Clean up statistics from any old runs
	if os.path.exists(workspace_folder):
	    print("Looks like you ran this before, so we need to cleanup those old files...")
	    shutil.rmtree(workspace_folder)
    
	os.makedirs(workspace_folder)
	workspace.ResetWorkspace(workspace_folder)

print("training data folder:" + data_folder)
print("workspace root folder:" + workspace_folder)

def AddInput(model, batch_size, db, db_type):
    ### load the data from db - Method 1 using brew
    #data_uint8, label = brew.db_input(
    #    model,
    #    blobs_out=["data_uint8", "label"],
    #    batch_size=batch_size,
    #    db=db,
    #    db_type=db_type,
    #)
    ### load the data from db - Method 2 using TensorProtosDB
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label
    
# Function to construct a MLP neural network
# The input 'model' is a model helper and 'data' is the input data blob's name
def AddMLPModel(model, data):
    size = 28 * 28 * 1
    sizes = [size, size * 2, size * 2, 10]
    layer = data
    for i in range(len(sizes) - 1):
        layer = brew.fc(model, layer, 'dense_{}'.format(i), dim_in=sizes[i], dim_out=sizes[i + 1])
        layer = brew.relu(model, layer, 'relu_{}'.format(i))
    softmax = brew.softmax(model, layer, 'softmax')
    return softmax    
    
def AddLeNetModel(model, data):
    '''
    This part is the standard LeNet model: from data to the softmax prediction.
    
    For each convolutional layer we specify dim_in - number of input channels
    and dim_out - number or output channels. Also each Conv and MaxPool layer changes the
    image size. For example, kernel of size 5 reduces each side of an image by 4.

    While when we have kernel and stride sizes equal 2 in a MaxPool layer, it divides
    each side in half.
    '''
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    # Here, the data is flattened from a tensor of dimension 50x4x4 to a vector of length 50*4*4
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    relu3 = brew.relu(model, fc3, 'relu3')
    # Last FC Layer
    pred = brew.fc(model, relu3, 'pred', dim_in=500, dim_out=10)
    # Softmax Layer
    softmax = brew.softmax(model, pred, 'softmax')
    
    return softmax



def AddModel(model, data):
    if USE_LENET_MODEL:
        return AddLeNetModel(model, data)
    else:
        return AddMLPModel(model, data)



def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy

def AddTrainingOperators(model, softmax, label):
    """Adds training operators to the model."""
    # Compute cross entropy between softmax scores and labels
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # Compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # Track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # Use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # Specify the optimization algorithm
    optimizer.build_sgd(
        model,
        base_learning_rate=0.1,
        policy="step",
        stepsize=1,
        gamma=0.999,
    )    
    
def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.
    
    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """    
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     workspace_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.    
    
def draw_graph(net, file_path):
    print("Started drawing...", file_path)
    graph = net_drawer.GetPydotGraph(net, rankdir="LR")

    filename, fileext = os.path.splitext(file_path)

    try:
        if fileext == ".jpg":
           graph.write_jpg(file_path)
        elif fileext == ".png":
           graph.write_png(file_path)
        else:
           print("unsupported extension")
    except Exception as e:
        print('Error when writing graph to image {}'.format(e))
    print("Finished drawing")    
   
if args.train: 
	#### Train Model
	# Specify the data will be input in NCHW order
	#  (i.e. [batch_size, num_channels, height, width])
	arg_scope = {"order": "NCHW"}
	# Create the model helper for the train model
	train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
	# Specify the input is from the train lmdb
	data, label = AddInput(
	    train_model, batch_size=64,
	    db=os.path.join(data_folder, 'mnist-train-nchw-lmdb'),
	    db_type='lmdb')
	# Add the model definition (fc layers, conv layers, softmax, etc.)
	softmax = AddModel(train_model, data)
	# Add training operators, specify loss function and optimization algorithm
	AddTrainingOperators(train_model, softmax, label)
	# Add bookkeeping operators to save stats from training
	AddBookkeepingOperators(train_model)

	#### Testing model. 
	# We will set the batch size to 100, so that the testing
	#   pass is 100 iterations (10,000 images in total).
	#   For the testing model, we need the data input part, the main AddModel
	#   part, and an accuracy part. Note that init_params is set False because
	#   we will be using the parameters obtained from the train model which will
	#   already be in the workspace.
	test_model = model_helper.ModelHelper(
	    name="mnist_test", arg_scope=arg_scope, init_params=False)
	data, label = AddInput(
	    test_model, batch_size=100,
	    db=os.path.join(data_folder, 'mnist-test-nchw-lmdb'),
	    db_type='lmdb')
	softmax = AddModel(test_model, data)
	AddAccuracy(test_model, softmax, label)

	#### Deployment model. 
	# We simply need the main AddModel part.
	deploy_model = model_helper.ModelHelper(
	    name="mnist_deploy", arg_scope=arg_scope, init_params=False)
	AddModel(deploy_model, "data")    

	draw_graph(train_model.net, os.path.join(figure_folder, "train_model_def.jpg"))

	print("*******train_model.net.Proto()*******\n")
	#print(str(train_model.net.Proto())[:400] + '\n...')
	print(str(train_model.net.Proto()))
	print("\n*******train_model.param_init_net.Proto()*******\n")
	#print(str(train_model.param_init_net.Proto())[:400] + '\n...')
	print(str(train_model.param_init_net.Proto()))

	# The parameter initialization network only needs to be run once.
	# Now all the parameter blobs are initialized in the workspace.
	workspace.RunNetOnce(train_model.param_init_net)

	# Creating an actual network as a C++ object in memory.
	#   We need this as the object is going to be used a lot
	#   so we avoid creating an object every single time it is used.
	# overwrite=True allows you to run this cell several times and avoid errors
	workspace.CreateNet(train_model.net, overwrite=True)

	# Set the iterations number and track the accuracy & loss
	accuracy = np.zeros(args.total_iters)
	loss = np.zeros(args.total_iters)

	# MAIN TRAINING LOOP!
	# Now, we will manually run the network for 200 iterations. 
	for i in range(args.total_iters):
	    workspace.RunNet(train_model.net)
	    accuracy[i] = workspace.blobs['accuracy']
	    loss[i] = workspace.blobs['loss']
	    # Check the accuracy and loss every so often
	    if i % 25 == 0:
	        print("Iter: {}, Loss: {}, Accuracy: {}".format(i,loss[i],accuracy[i]))

	# After the execution is done, let's plot the values.
	pyplot.plot(loss, 'b')
	pyplot.plot(accuracy, 'r')
	pyplot.title("Summary of Training Run")
	pyplot.xlabel("Iteration")
	pyplot.legend(('Loss', 'Accuracy'), loc='upper right')
	pyplot.savefig(os.path.join(figure_folder, 'mnist_training.png'))

	### Let's look at some of the training data
	pyplot.figure()
	pyplot.title("Training Data Sample")
	# Grab the most recent data blob (i.e. batch) from the workspace
	data = workspace.FetchBlob('data')
	# Use visualize module to show the examples from the last batch that was fed to the model
	_ = visualize.NCHW.ShowMultiple(data)
	pyplot.savefig(os.path.join(figure_folder, 'mnist_training_data.png'))

	### Let's visualize the softmax result
	pyplot.figure()
	pyplot.title('Softmax Prediction for the first image above')
	pyplot.ylabel('Confidence')
	pyplot.xlabel('Label')
	# Grab and visualize the softmax blob for the batch we just visualized. Since batch size
	#  is 64, the softmax blob contains 64 vectors, one for each image in the batch. To grab
	#  the vector for the first image, we can simply index the fetched softmax blob at zero.
	softmax = workspace.FetchBlob('softmax')
	_ = pyplot.plot(softmax[0], 'ro')
	pyplot.savefig(os.path.join(figure_folder, 'mnist_softmax.png'))



	if USE_LENET_MODEL:
	    pyplot.figure()
	    pyplot.title("Conv1 Output Feature Maps for Most Recent Mini-batch")
	    # Grab the output feature maps of conv1. Change this to conv2 in order to look into the second one.
	    #  Remember, early convolutional layers tend to learn human-interpretable features but later conv
	    #  layers work with highly-abstract representations. For this reason, it may be harder to understand
	    #  features of the later conv layers.
	    conv = workspace.FetchBlob('conv1')
	   
	    # We can look into any channel. Think of it as a feature model learned.
	    # In this case we look into the 5th channel. Play with other channels to see other features
	    conv = conv[:,[5],:,:]
	    _ = visualize.NCHW.ShowMultiple(conv)
	    pyplot.savefig(os.path.join(figure_folder, 'mnist_conv.png'))

	    
	# param_init_net here will only create a data reader
	# Other parameters won't be re-created because we selected init_params=False before
	workspace.RunNetOnce(test_model.param_init_net)
	workspace.CreateNet(test_model.net, overwrite=True)

	# Testing Loop 
	test_accuracy = np.zeros(100)
	for i in range(100):
	    # Run a forward pass of the net on the current batch
	    workspace.RunNet(test_model.net)
	    # Collect the batch accuracy from the workspace
	    test_accuracy[i] = workspace.FetchBlob('accuracy')
	    
	# After the execution is done, let's plot the accuracy values
	pyplot.figure()
	pyplot.plot(test_accuracy, 'r')
	pyplot.title('Accuracy over test batches.')
	pyplot.savefig(os.path.join(figure_folder, 'mnist_test_accuracy.png'))
	print('test_accuracy: %f' % test_accuracy.mean())

	# construct the model to be exported
	# the inputs/outputs of the model are manually specified.
	pe_meta = pe.PredictorExportMeta(
	    predict_net=deploy_model.net.Proto(),
	    parameters=[str(b) for b in deploy_model.params], 
	    inputs=["data"],
	    outputs=["softmax"],
	)

	# save the model to a file. Use minidb as the file format
	pe.save_to_db("minidb", os.path.join(workspace_folder, "mnist_model.minidb"), pe_meta)
	print("Deploy model saved to: " + workspace_folder + "/mnist_model.minidb")


	# Grab and display the last data batch used before we scratch the workspace. This purely for our convenience...
	blob = workspace.FetchBlob("data")
	pyplot.figure()
	pyplot.title("Batch of Testing Data")
	_ = visualize.NCHW.ShowMultiple(blob)
	pyplot.savefig(os.path.join(figure_folder, 'mnist_testing_data.png'))

def get_image_blob(im, target_size):
    '''
    im is an UINT8 image with the shape Height, Width, Color channel
    Returns:
       blob (ndarray): a data blob holding an image
    '''
    # Convert to core.DataType.FLOAT (np.float32) 
    im = im.astype(np.float32, copy=False)
    im_shape = im.shape
    im_scale_x = float(target_size) / float(im_shape[1])
    im_scale_y = float(target_size) / float(im_shape[0])
    # Rescale to the specified target size
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=cv2.INTER_LINEAR
    )
    # scale data from [0,255] down to [0,1]
    im /= 256.0    
    
    """Convert an image into a network input. 
    float32 numpy ndarray format
    Output is a 4D NCHW tensor of the image
    """
    blob = np.zeros((1, 1, im.shape[0], im.shape[1]), dtype=np.float32)
    # convert image to monochrome before copying it into the blob
    blob[0, 0, 0:im.shape[0], 0:im.shape[1]] = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    # Axis order are: (batch elem, channel, height, width)
    return blob

if args.test:
	# reset the workspace, to make sure the model is actually loaded
	workspace.ResetWorkspace(workspace_folder)

	# verify that all blobs from training are destroyed. 
	print("The blobs in the workspace after reset: {}".format(workspace.Blobs()))

	# load the predict net
	predict_net = pe.prepare_prediction_net(os.path.join(workspace_folder, "mnist_model.minidb"), "minidb")
	
	print("*******predict_net.Proto()*******\n")
	print(str(predict_net.Proto()))

	draw_graph(predict_net, os.path.join(figure_folder, "predict_model_def.jpg"))	

	# verify that blobs are loaded back
	print("The blobs in the workspace after loading the model: {}".format(workspace.Blobs()))

	# feed an image to the loaded model
	im = cv2.imread(args.image_path)
	blob = get_image_blob(im, 28)
	workspace.FeedBlob("data", blob)

	# predict
	workspace.RunNetOnce(predict_net)
	softmax = workspace.FetchBlob("softmax")

	print("Shape of softmax: ",softmax.shape)

	# Quick way to get the top-1 prediction result
	# Squeeze out the unnecessary axis. This returns a 1-D array of length 10
	# Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
	curr_pred, curr_conf = max(enumerate(softmax[0]), key=operator.itemgetter(1))
	print("Prediction: ", curr_pred)
	print("Confidence: ", curr_conf)

	# the first letter should be predicted correctly
	pyplot.figure()
	pyplot.title('Prediction for the first image')
	pyplot.ylabel('Confidence')
	pyplot.xlabel('Label')
	_ = pyplot.plot(softmax[0], 'ro')
	pyplot.savefig(os.path.join(figure_folder, 'mnist_prediction.png'))

	# Grab and display the last data batch used before we scratch the workspace. This purely for our convenience...
	blob = workspace.FetchBlob("data")
	pyplot.figure()
	pyplot.title("Test Image")
	_ = visualize.NCHW.ShowMultiple(blob)
	pyplot.savefig(os.path.join(figure_folder, 'mnist_test_image.png'))
