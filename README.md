# sklearn_scripts
Script for running MLP and KRR using the python package scikitlearn.

Input files need to be in npy format.
Feature files can be in sparse matrix (npz) format.
One datapoint per row, one feature per column.

In order to change a file from csv to numpy, call

csv_to_numpy.py csvfilename

Run the script with

python3 [run type] [xfile] [yfile] [krr or mlp]

e.g.

python3 run features.npy labels.npy krr

There are 4 runtypes: run, param, size, psize
	run: 	one run for specified training set size, with fixed hyperparamters
	param: 	grid run for specified training set size, with CV hyperparamters
	size: 	same as run with a list of different training set sizes
	psize: 	param and size combined. A part of the training set is used for CV of hyperparameters
		(can be changed with paramset_size) 

The next two arguments need to be specified.
	xfile:	features of datapoints to train on
	yfile:	labels of the datapoints

The 4th positional argument needs to be specified
	krr: 	kernel ridge regression
	mlp:	multi-layer perceptron or feedforward neural network

Apart from that, you can choose how the training-test-set separation is controlled:
	default: 	random selection
			
	--ids: 		featureids.npy labelids
			training and test set are separated by id 
			which is the order of appearance in the input files
			They need to be specified in two numpy format
	--testfiles:	features labels
			training and test set are separated by files
			give two files in numpy format (featurefile can be in sparse matrix format "npz")
			With this option, 4 data files are read
			xfile, yfile	become trainingset
			features, labels become testset
			

details can be changed in sklearn_krr_mlp.py
(alpha, gamma, kernels, hidden layers, ...)

By default alpha=1e-3 and gamma=1e-5 are used.
They can be changed in the INPUT section in sklearn_krr_mlp.py for the specific runtype

# FPS
Farthest point sampling
