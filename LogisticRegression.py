import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('Datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('Datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(dim):
    '''
    Desc: used to initialize the weight and bias matrix
    Input: number of dimensions
    Output:
    1.  w -> Weight Matrix
    2.  b -> Bias 
    '''
    w = np.zeros(dim).reshape(dim, 1)
    b = 0

    return w, b


def sigmoid(z):
    '''
    Desc: used to define the sigmoid activation function
    Input: z = w.X + b
    Output: sigmoid(z)
    '''
    sigmoid = 1 / (1 + np.exp(-z)) 

    return sigmoid


def propogate(w, b, X, Y):
    '''
    Desc: code to define the forward and backward propogation
    Input: 
    1.  w -> Weight Matrix
    2.  X -> Input Data
    3.  b -> Bias
    4.  Y -> Correct / Actual output label

    Output:
    1.  dW -> Gradient of W
    2.  db -> Gradient of b
    3.  cost -> Negative log liklihood cost for logistic regression
    '''
    ## Forward Propogation

    # initializing varialble m with no. of samples in X
    m = X.shape[1]

    # calculating the activation function
    A = sigmoid(np.dot(w.T, X) + b)

    # calculating negative log liklihood cost for logistic regression
    cost = - (1 / m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))

    ## backward Propogation
    dw = np.dot(X, (A-Y).T) / m
    db = np.sum(A - Y) / m

    grads = {"dw" : dw, "db" : db}
    cost = np.squeeze(cost)

    return grads, cost


def fit(w, b, X, Y, num_iterations, alpha, print_cost = False):
    '''
    Desc: Fit's the best values of parameters w and b on the given input data
    
    Input: 
    1.  w -> Initial Weight Matrix
    2.  b -> Initial Bias
    3.  X -> Training Data
    4.  Y -> Training Label
    5.  num_iterations -> No. of times to run gradient descent for optimization
    6.  alpha -> Learning Rate
    7.  print_cost -> if set to "True" will print cost function every 100 iterations
    
    Output:
    1.  w -> Optimized Weights
    2.  b -> Optimized Bias
    '''
    costs = []

    # Running gradient descent for num of iterations
    for i in range(num_iterations):
        grads, cost = propogate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        # updating the values of w and b
        w = w - (alpha * dw)
        b = b - (alpha * db)

        if i % 100 == 0:
            costs.append(cost)

        # Printing cost every 100 iterations if print_cost is set to "True"
        if print_cost and i % 100 == 0:
            print ("Cost after {} iteration is: {}".format(i, round(cost, 6)))
        
        params = {"w" : w, "b" : b}
        grads = {"dw" : dw, "db" : db}

    return params, grads, costs


def predict(w, b, X):
    '''
    Desc: used for making predictions on test dataset using learned parameters of w & b
    
    Input: 
    1.  w -> Optimized Weights
    2.  b -> Optimized Bias
    3.  X -> Test Data(num_px * num_px * 3, num_examples)

    Output:
    1.  Y_prediction -> Predicted Labels
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T,X) + b)
    
    # assign Y_prediction to be 1 if activation value (A > 0.5) else assign Y_prediction to be 0
    Y_prediction = np.where(A > 0.5, 1, 0)

    return Y_prediction