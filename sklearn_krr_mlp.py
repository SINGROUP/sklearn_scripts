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

def load_data(is_sparse, x_datafile, y_datafile):
    # Load all the data
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
    return x_data, y_data, ids, is_mean

def split_and_scale_data(x_data, y_data, ids, sample_size, is_mean):
    # split set 
    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(x_data, y_data, ids, test_size = 1 - sample_size)
    
    # Scale
    scaler = StandardScaler(with_mean=is_mean)  
        # fit only on training data
    scaler.fit(x_train)  
    x_train = scaler.transform(x_train)  
        # apply same transformation to test data
    x_test = scaler.transform(x_test)  

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

def ml_param_scan(x_datafile, y_datafile, alpha_list= np.logspace(-1, -8, 8), 
    gamma_list = np.logspace(-2, -10, 9), kernel_list = ['rbf'], 
    layer_list = [(40,40,40)], learning_rate_list = [0.001], is_sparse = False,
    sample_size=0.1, ml_method = "krr"):
    print('model ' + ml_method)
    # Load all the data
    x_data, y_data, ids, is_mean = load_data(is_sparse, x_datafile, y_datafile)
    
    #split set, scale features x
    x_train, x_test, y_train, y_test, ids_train, ids_test = split_and_scale_data(x_data, y_data, ids, 1 - sample_size , is_mean)

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


def ml_run(x_datafile, y_datafile, 
    alpha0=1, gamma0=1, kernel0 = 'rbf', learning_rate_init0 = 0.001, 
    hidden_layer_sizes0=(80, 80, 80), is_sparse = False, 
    sample_size=0.1, ml_method = "krr"):
    print('model ' + ml_method)

    # Load all the data
    x_data, y_data, ids, is_mean = load_data(is_sparse, x_datafile, y_datafile)
    
    #split set, scale features x
    x_train, x_test, y_train, y_test, ids_train, ids_test = split_and_scale_data(x_data, y_data, ids, sample_size, is_mean)

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

def ml_size_scan(x_datafile, y_datafile, alpha0=1, gamma0=1, 
    kernel0 = 'rbf', learning_rate_init0 = 0.001, 
    hidden_layer_sizes0=(80, 80, 80), is_sparse = False, 
    sample_size_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    ml_method = "krr"):

    # Load all the data
    x_data, y_data, ids, is_mean = load_data(is_sparse, x_datafile, y_datafile)
    
    #split set, scale features x
    max_sample_size = max(sample_size_list)
    ax_train, x_test, ay_train, y_test, aids_train, ids_test = split_and_scale_data(x_data, y_data, ids, max_sample_size, is_mean)

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

def ml_param_size_scan(x_datafile, y_datafile, alpha_list= np.logspace(-1, -8, 8), 
    gamma_list = np.logspace(-2, -10, 9), kernel_list = ['rbf'], 
    layer_list = [(40,40,40)], learning_rate_list = [0.001], is_sparse = False, 
    sample_size_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    ml_method = "krr", testset_size=0.1):
    print('model ' + ml_method)

    # Load all the data
    x_data, y_data, ids, is_mean = load_data(is_sparse, x_datafile, y_datafile)
    
    #split set, scale features x
    max_sample_size = max(sample_size_list)
    ax_train, ax_test, ay_train, ay_test, aids_train, aids_test = split_and_scale_data(x_data, y_data, ids, max_sample_size, is_mean)



    # search for optimal learner parameters
    # reduce set
    x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split(ax_train, ay_train, aids_train, test_size = testset_size,)

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
    write_output(learner, max_sample_size * (1 - testset_size), ml_method, mae, mse, "param", ids_test, y_test, y_pred, ids_train, y_train, train_y_pred)


    # use above found best parameters
    x_test = ax_test
    y_test = ay_test
    ids_test = aids_test
    paramlearner = learner

    for sample_size in sample_size_list:
        print("Learning from part of the training set:", sample_size)
        # reduce set ("test set" is the part of the training set that is used)
        x_dump, x_train, y_dump, y_train, ids_dump, ids_train = train_test_split(ax_train, ay_train, aids_train, test_size = sample_size,)



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

parser.add_argument('arguments', metavar='cla', type=str, nargs='+',
                   help='[Runtype] [descriptor] [predictor] [krr or mlp]')

args = parser.parse_args()
print("Arguments passed:")
print(args.arguments)

runtype = args.arguments[0]
descfile = args.arguments[1]
predfile = args.arguments[2]
ML_METHOD = args.arguments[3]
if "npz" in descfile:
    IS_SPARSE = True
else:
    IS_SPARSE = False


### PROCESS ###

if runtype == "param":
    ml_param_scan(descfile, predfile,
        alpha_list= np.logspace(-1, -9, 9),
        gamma_list = np.logspace(-1, -9, 9), kernel_list = ['rbf'], 
        layer_list = [(40,40,40)], learning_rate_list = [0.001], 
        is_sparse = IS_SPARSE,
        sample_size=0.1, ml_method = ML_METHOD)

elif runtype == "run":
    ml_run(descfile, predfile, 
        alpha0=1e-4, gamma0=1e-03, kernel0 = 'rbf', 
        learning_rate_init0 = 0.001, hidden_layer_sizes0=(80, 80, 80), 
        is_sparse = IS_SPARSE, sample_size=0.1, ml_method = ML_METHOD)

elif runtype == "size":
    ml_size_scan(descfile, predfile,  
        alpha0=1e-9, gamma0=1e-10, kernel0 = 'rbf', 
        learning_rate_init0 = 0.001, hidden_layer_sizes0=(80, 80, 80), 
        is_sparse = IS_SPARSE, 
        #sample_size_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
        sample_size_list = [0.005, 0.01, 0.03, 0.05], 
        ml_method = ML_METHOD)

elif runtype == "psize":
    ml_param_size_scan(descfile, predfile, alpha_list= np.logspace(-1, -8, 8), 
    gamma_list = np.logspace(-2, -10, 9), kernel_list = ['rbf'], 
    layer_list = [(40,40,40)], learning_rate_list = [0.001], is_sparse = IS_SPARSE, 
    #sample_size_list = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9,], 
    sample_size_list = [0.005, 0.01, 0.03], 
    ml_method = ML_METHOD, testset_size=0.5)

else:
    print("First argument not understood:")
    print("Usage: python3 SCRIPTNAME [param, run, size or psize] [features] [labels] [krr or mlp] ")
    exit(1)


### OUTPUT ###

