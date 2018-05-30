import numpy as np
from scipy.sparse import load_npz
from sklearn import datasets, linear_model,metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import time, sys
import argparse

### DEFINE and INPUT ###

def load_data(x_datafile, y_datafile, ids, testfiles):
    # check if matrix is sparse by file ending
    if "npz" in descfile:
        is_sparse = True
    else:
        is_sparse = False

    ids_train = None
    ids_test = None
    x_test = None
    y_test = None
    # Load all the data
    # training and test is split by ID
    if ids:
        ids_train = np.load(ids[0])
        ids_test = np.load(ids[1])
    # training and test is split by file
    elif testfiles:
        if is_sparse:
            x_test = load_npz(testfiles[0])
            is_mean = False
        else:
            x_test = np.load(testfiles[0])
            is_mean = True
        y_test = np.load(testfiles[1])
        # discard other columns except first
        if len(y_test.shape) > 1:
            y_test = y_test[:, 0]
        # create pythonic ids
        ids_test = np.arange(len(y_test))

    if is_sparse:
        x_data = load_npz(x_datafile)
        is_mean = False
    else:
        x_data = np.load(x_datafile)
        is_mean = True
    y_data = np.load(y_datafile)
    # discard other columns except first
    if len(y_data.shape) > 1:
        y_data = y_data[:, 0]
    # create pythonic ids
    ids = np.arange(len(y_data))
    return x_data, y_data, ids, is_mean, is_sparse, ids_train, ids_test, x_test, y_test

def split_data(x_data, y_data, ids_data, sample_size, is_mean, is_sparse, ids, ids_train, ids_test):
    # split set 
    if ids:
        if is_sparse:
            x_data = x_data.tolil()
        
        x_train = x_data[ids_train]
        x_test = x_data[ids_test]
        y_train = y_data[ids_train]
        y_test = y_data[ids_test]

        if is_sparse:
            x_train = x_train.tocsr()
            x_test = x_test.tocsr()
    else:
        x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(x_data, y_data, ids_data, test_size = 1 - sample_size)
    
    return x_train, x_test, y_train, y_test, ids_train, ids_test

def scale_data(is_mean, x_train, x_test):
    # Scale
    #scaler = StandardScaler(with_mean=is_mean)  
        # fit only on training data
    #scaler.fit(x_train)  
    #x_train = scaler.transform(x_train)  
        # apply same transformation to test data
    #x_test = scaler.transform(x_test)  
    return x_train, x_test

def load_split_scale_data(x_datafile, y_datafile, ids, testfiles,
    sample_size):
    # load
    x_data, y_data, ids_data, is_mean, is_sparse, ids_train, ids_test, x_test, y_test = load_data(
        x_datafile, y_datafile, ids, testfiles)
    # split
    if testfiles:
        x_train = x_data
        y_train = y_data
        x_test = x_test
        y_test = y_test
        ids_train = ids_data
        ids_test = ids_test
    else:
        x_train, x_test, y_train, y_test, ids_train, ids_test = split_data(
            x_data, y_data, ids_data, sample_size, is_mean, is_sparse, ids, ids_train, ids_test)
        # scale
    x_train, x_test = scale_data(is_mean, x_train, x_test)
    return x_train, x_test, y_train, y_test, ids_train, ids_test

def predict_and_error(learner, x_test, x_train, y_test):

    y_pred = learner.predict(x_test)

    # run also on training set
    train_y_pred = learner.predict(x_train)

    # errors
    mae = np.absolute(y_pred - y_test)
    mse = mae ** 2

    mae = np.mean(mae)
    mse = np.mean(mse)
    return mae, mse, y_pred, train_y_pred, learner


def write_output(learner, sample_size, ml_method, mae, mse, runtype, ids_test, y_test, y_pred, ids_train, y_train, train_y_pred):
    ### OUTPUT ###
    # y_test vs y_predict
    y_tmp = np.array([ids_test, y_test, y_pred])
    y_compare = np.transpose(y_tmp)

    np.savetxt(ml_method + str("_") + runtype + "_size" + str(sample_size) + ".predictions", y_compare, 
        header = "###ids_test   y_test    y_pred")

    # also y_train vs. y_pred_train
    y_tmp = np.array([ids_train, y_train, train_y_pred])
    y_compare = np.transpose(y_tmp)

    np.savetxt(ml_method + str("_") + runtype + "_size" + str(sample_size) + ".trainset_predictions", y_compare, 
        header = "###ids_train   y_train    train_y_pred")

    with open(ml_method + str("_") + runtype + "_size" + str(sample_size) + ".out", "w") as f:
        f.write("MAE " + str(mae) + "\n")
        f.write("MSE " + str(mse) + "\n")
        f.write("\n")
        if runtype == "param":
            f.write("Best parameters of " + ml_method + ": \n")
            f.write(str(learner.best_params_) + "\n")
            f.write("Errors of best parameters: \n")



            f.write("Grid scores on validation set:" + "\n")
            f.write("\n")
            means = learner.cv_results_['mean_test_score']
            stds = learner.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, learner.cv_results_['params']):
                f.write("%0.4f (+/-%0.04f) for %r"
                      % (mean, std * 2, params) + "\n")
            f.write("\n")
            f.write("Grid scores on train set:" + "\n")
            f.write("\n")        
            means = learner.cv_results_['mean_train_score']
            stds = learner.cv_results_['std_train_score']        
            for mean, std, params in zip(means, stds, learner.cv_results_['params']):
                f.write("%0.4f (+/-%0.04f) for %r"
                      % (mean, std * 2, params) + "\n")

    return None

def ml_param_scan(x_datafile, y_datafile, ids, testfiles, 
    alpha_list= np.logspace(-1, -8, 8), 
    gamma_list = np.logspace(-2, -10, 9), kernel_list = ['rbf'], 
    layer_list = [(40,40,40)], learning_rate_list = [0.001],
    sample_size=0.1, ml_method = "krr"):
    print('model ' + ml_method)
    # load, split and scale data
    x_train, x_test, y_train, y_test, ids_train, ids_test = load_split_scale_data(x_datafile, y_datafile, ids, testfiles,
    sample_size)

    if ml_method == "krr":
        # Create kernel linear ridge regression object
        learner = GridSearchCV(KernelRidge(kernel='rbf'), n_jobs = 8, cv=5,
                      param_grid={"alpha": alpha_list, "gamma": gamma_list, 
                      "kernel": kernel_list}, scoring = 'neg_mean_absolute_error')

    elif ml_method == "mlp":
        # Create Multi-Layer Perceptron object
        learner = GridSearchCV(MLPRegressor(hidden_layer_sizes=(40,40,40), max_iter=1600, 
            alpha= 0.001, learning_rate_init= 0.001), n_jobs = 8, cv=5,
                      param_grid={"alpha": alpha_list, "learning_rate_init": learning_rate_list, 
                      "hidden_layer_sizes": layer_list}, scoring = 'neg_mean_absolute_error')
    else:
        print("ML method unknown. Exiting.")
        exit(1)
    t_ml0 = time.time()
    learner.fit(x_train, y_train)
    t_ml1 = time.time()
    print("ml time", str(t_ml1 - t_ml0))

    # getting best parameters
    learner_best = learner.best_estimator_

    mae, mse, y_pred, train_y_pred, learner_best = predict_and_error(learner_best, x_test, x_train, y_test)

    ### OUTPUT ###
    write_output(learner, sample_size, ml_method, mae, mse, "param", ids_test, y_test, y_pred, ids_train, y_train, train_y_pred)

    return learner.best_params_


def ml_run(x_datafile, y_datafile, ids, testfiles, 
    alpha0=1, gamma0=1, kernel0 = 'rbf', learning_rate_init0 = 0.001, 
    hidden_layer_sizes0=(80, 80, 80), 
    sample_size=0.1, ml_method = "krr"):
    print('model ' + ml_method)
    # load, split and scale data
    x_train, x_test, y_train, y_test, ids_train, ids_test = load_split_scale_data(x_datafile, y_datafile, ids, testfiles,
    sample_size)

    if ml_method == "krr":
        # Create kernel linear ridge regression object
        learner = KernelRidge(alpha = alpha0, coef0=1, degree=3, 
            gamma= gamma0, kernel = kernel0, kernel_params=None)

    elif ml_method == "mlp":
        # Create Multi-Layer Perceptron object
        learner = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes0, max_iter=1600, 
            alpha= alpha0, learning_rate_init= learning_rate_init0)

    else:
        print("ML method unknown. Exiting.")
        exit(1)

    t_ml0 = time.time()
    learner.fit(x_train, y_train)
    t_ml1 = time.time()
    print("ml time", str(t_ml1 - t_ml0))

    mae, mse, y_pred, train_y_pred, learner = predict_and_error(learner, x_test, x_train, y_test)

    ### OUTPUT ###
    write_output(learner, sample_size, ml_method, mae, mse, "run", ids_test, y_test, y_pred, ids_train, y_train, train_y_pred)

    return None

def ml_size_scan(x_datafile, y_datafile, ids, testfiles, 
    alpha0=1, gamma0=1, 
    kernel0 = 'rbf', learning_rate_init0 = 0.001, 
    hidden_layer_sizes0=(80, 80, 80),
    sample_size_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    ml_method = "krr"):    
    max_sample_size = max(sample_size_list)
    ax_train, x_test, ay_train, y_test, aids_train, ids_test = load_split_scale_data(x_datafile, y_datafile, ids, testfiles,
    max_sample_size)

    for sample_size in sample_size_list:
        ratio = float(sample_size) / float(max_sample_size)
        if ratio > 0.999:
            x_train = ax_train
            y_train = ay_train
            ids_train = aids_train
        else:
            # reduce set
            x_dump, x_train, y_dump, y_train, ids_dump, ids_train = train_test_split(ax_train, ay_train, aids_train, test_size = ratio,)

        if ml_method == "krr":
            # Create kernel linear ridge regression object
            learner = KernelRidge(alpha = alpha0, coef0=1, degree=3, 
                gamma= gamma0, kernel = kernel0, kernel_params=None)
        elif ml_method == "mlp":
            # Create Multi-Layer Perceptron object
            learner = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes0, max_iter=1600, 
                alpha= alpha0, learning_rate_init= learning_rate_init0)
        else:
            print("ML method unknown. Exiting.")
            exit(1)

        t_ml0 = time.time()
        learner.fit(x_train, y_train)
        t_ml1 = time.time()
        print("ml time", str(t_ml1 - t_ml0))

        mae, mse, y_pred, train_y_pred, learner = predict_and_error(learner, x_test, x_train, y_test)

        ### OUTPUT ###
        write_output(learner, sample_size, ml_method, mae, mse, "size", ids_test, y_test, y_pred, ids_train, y_train, train_y_pred)

    return None

def ml_param_size_scan(x_datafile, y_datafile, ids, testfiles, 
    alpha_list= np.logspace(-1, -8, 8), 
    gamma_list = np.logspace(-2, -10, 9), kernel_list = ['rbf'], 
    layer_list = [(40,40,40)], learning_rate_list = [0.001],
    sample_size_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    ml_method = "krr", paramset_size=0.1):
    print('model ' + ml_method)
    max_sample_size = max(sample_size_list)
    # load, split and scale data
    ax_train, ax_test, ay_train, ay_test, aids_train, aids_test = load_split_scale_data(x_datafile, y_datafile, ids, testfiles,
    max_sample_size)

    # search for optimal learner parameters
    # reduce set
    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(ax_train, ay_train, aids_train, test_size = 1 - paramset_size,)

    if ml_method == "krr":
        # Create kernel linear ridge regression object
        learner = GridSearchCV(KernelRidge(kernel='rbf'), n_jobs = 8, cv=5,
                      param_grid={"alpha": alpha_list, "gamma": gamma_list, 
                      "kernel": kernel_list}, scoring = 'neg_mean_absolute_error')

    elif ml_method == "mlp":
        # Create Multi-Layer Perceptron object
        learner = GridSearchCV(MLPRegressor(hidden_layer_sizes=(40,40,40), max_iter=1600, 
            alpha= 0.001, learning_rate_init= 0.001), n_jobs = 8, cv=5,
                      param_grid={"alpha": alpha_list, "learning_rate_init": learning_rate_list, 
                      "hidden_layer_sizes": layer_list}, scoring = 'neg_mean_absolute_error')
    else:
        print("ML method unknown. Exiting.")
        exit(1)
    t_ml0 = time.time()
    learner.fit(x_train, y_train)
    t_ml1 = time.time()
    print("ml time", str(t_ml1 - t_ml0))

    # getting best parameters
    learner_best = learner.best_estimator_

    mae, mse, y_pred, train_y_pred, learner_best = predict_and_error(learner_best, x_test, x_train, y_test)

    ### OUTPUT ###
    write_output(learner, max_sample_size *  paramset_size, ml_method, mae, mse, "param", ids_test, y_test, y_pred, ids_train, y_train, train_y_pred)


    # use above found best parameters
    x_test = ax_test
    y_test = ay_test
    ids_test = aids_test
    paramlearner = learner

    for sample_size in sample_size_list:
        ratio = float(sample_size) / float(max_sample_size)
        if ratio > 0.999:
            x_train = ax_train
            y_train = ay_train
            ids_train = aids_train
        else:
            # reduce set ("test set" is the part of the training set that is used)
            x_dump, x_train, y_dump, y_train, ids_dump, ids_train = train_test_split(ax_train, ay_train, aids_train, test_size = ratio,)

        if ml_method == "krr":
            # Create kernel linear ridge regression object

            alpha0 = paramlearner.best_params_["alpha"]
            gamma0 = paramlearner.best_params_["gamma"]
            kernel0 = paramlearner.best_params_["kernel"]

            learner = KernelRidge(alpha = alpha0, coef0=1, degree=3, 
                gamma= gamma0, kernel = kernel0, kernel_params=None)
        elif ml_method == "mlp":
            # Create Multi-Layer Perceptron object

            hidden_layer_sizes0 = paramlearner.best_params_["hidden_layer_sizes"]
            alpha0 = paramlearner.best_params_["alpha"]
            learning_rate_init0 = paramlearner.best_params_["learning_rate_init"]

            learner = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes0, max_iter=1600, 
                alpha= alpha0, learning_rate_init= learning_rate_init0)
        else:
            print("ML method unknown. Exiting.")
            exit(1)

        t_ml0 = time.time()
        learner.fit(x_train, y_train)
        t_ml1 = time.time()
        print("ml time", str(t_ml1 - t_ml0))

        mae, mse, y_pred, train_y_pred, learner = predict_and_error(learner, x_test, x_train, y_test)

        ### OUTPUT ###
        write_output(learner, sample_size, ml_method, mae, mse, "psize", ids_test, y_test, y_pred, ids_train, y_train, train_y_pred)
    return None

### INPUT ###


parser = argparse.ArgumentParser(description='Process runtype and filenames.')

parser.add_argument(dest='positionargs', metavar='cla', type=str, nargs='+',
                   help='[Runtype] [descriptor] [predictor] [krr or mlp]')
parser.add_argument('--ids', dest='ids', nargs=2, help='path to numpy arrays with indices to use as [training set] [test set]')
parser.add_argument('--testfiles', dest='testfiles', nargs=2, help='path to test set features and labels. features and labels for training set are given by positional argument')

args = parser.parse_args()
print("Arguments passed:")
print(args.positionargs)
if args.ids:
    print('ids for training set', args.ids[0])
    print('ids for test set', args.ids[1])
if args.testfiles:
    print('optional test set files', args.testfiles[0], args.testfiles[1])

runtype = args.positionargs[0]
descfile = args.positionargs[1]
predfile = args.positionargs[2]
ML_METHOD = args.positionargs[3]
ids = args.ids
testfiles = args.testfiles

### PROCESS ###

if runtype == "param":
    ml_param_scan(descfile, predfile, ids, testfiles,
        alpha_list= np.logspace(-1, -9, 9),
        gamma_list = np.logspace(-1, -9, 9), kernel_list = ['rbf'], 
        layer_list = [(40,40,40)], learning_rate_list = [0.001], 
        sample_size=0.1, ml_method = ML_METHOD)

elif runtype == "run":
    ml_run(descfile, predfile, ids, testfiles,
        alpha0=1e-4, gamma0=1e-03, kernel0 = 'rbf', 
        learning_rate_init0 = 0.001, hidden_layer_sizes0=(80, 80, 80), 
        sample_size=0.1, ml_method = ML_METHOD)

elif runtype == "size":
    ml_size_scan(descfile, predfile, ids, testfiles, 
        alpha0=1e-9, gamma0=1e-10, kernel0 = 'rbf', 
        learning_rate_init0 = 0.001, hidden_layer_sizes0=(80, 80, 80), 
        sample_size_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
        ml_method = ML_METHOD)

elif runtype == "psize":
    ml_param_size_scan(descfile, predfile, ids, testfiles,
    alpha_list= np.logspace(-1, -8, 8), 
    gamma_list = np.logspace(-2, -10, 9), kernel_list = ['rbf'], 
    layer_list = [(40,40,40)], learning_rate_list = [0.001], 
    sample_size_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,], 
    ml_method = ML_METHOD, paramset_size=0.11)

else:
    print("First argument not understood:")
    print("Usage: python3 SCRIPTNAME [param, run, size or psize] [features] [labels] [krr or mlp] ")
    exit(1)


### OUTPUT ###

