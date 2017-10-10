# sklearn_scripts
Scripts for running MLP and KRR using the python package scikitlearn.

Input files need to be in npy or sparse matrix (npz) format.
One datapoint per row, one feature per column.

In order to change a file from csv to numpy, call

csv_to_numpy.py csvfilename

Folder param_scan:

If you run
sbatch learn-debug.slrm featurefile labelfile

krr is run on 10% of the data. 5-fold CV is performed in order to scan alpha and gamma.

details can be changed in sklearn_krr_mlp.py

Folder scan_size:

If you run
sbatch learn-debug.slrm featurefile labelfile

krr is run with different training set sizes. By default alpha=1e-3 and gamma=1e-5 are used.
They can be changed in sklearn_krr_mlp.py

Folder train_test_by_id:
