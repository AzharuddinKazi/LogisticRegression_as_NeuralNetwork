import numpy as np
import h5py
import LogisticRegression as lr

def call_logistic(num_iterations = 2000, alpha = 0.5, print_cost = False):
    '''
    Desc : perform logistic regression on given dataset
    
    Input:
    1.  num_iterations -> num of times to run gradient descent 
    2.  alpha -> learning rate
    3.  print_cost -> prints the cost function every 100 iterations if set to "True"

    Output: Predictions
    '''
    ## Load data for pre-processing
    train_set_x_orig, Y_train, test_set_x_orig, Y_test, classes = lr.load_dataset()

    # flattening the train and test data, converting from 3D to 1D
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # standerdizing the data 
    X_train = train_set_x_flatten / 255
    X_test = test_set_x_flatten / 255

    ## building final model
    w, b = lr.initialize_parameters(X_train.shape[0])

    # fitting best values of w and b on data 
    parameters, grads, costs = lr.fit(w, b, X_train, Y_train, num_iterations, alpha, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = lr.predict(w, b, X_train)
    Y_prediction_test = lr.predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "alpha" : alpha,
         "num_iterations": num_iterations}

    return d

def main():
    '''
    Desc: calling different classifier functions
    '''
    call_logistic(num_iterations = 2000, alpha = 0.005, print_cost = True)

if __name__ == "__main__":
    main()