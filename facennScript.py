'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''
from math import sqrt
import numpy as np
from scipy.optimize import minimize
import pickle

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation

def sigmoid(z):
    return  1/(1+np.exp(-z))

# Replace this with your nnObjFunction implementation

def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    label_matrix = training_label

    #
    # Feedforward Propagation
    input_bias = np.ones((training_data.shape[0],1)) # create an bias
    training_data_bias = np.concatenate((training_data, input_bias), axis=1) # add bias to training data
    hiden_out = sigmoid(np.dot(training_data_bias, w1.T))  # 3.32 equtions 1 and 2
    hiden_bias = np.ones((1,hiden_out.T.shape[1])) # create an bias
    hiden_out_bias = np.concatenate((hiden_out.T, hiden_bias), axis=0).T  # add bias to hiden_out data
    net_out = sigmoid(np.dot(hiden_out_bias,w2.T)) # 3.32 eqution 3 and 4, feed forward is complete.

    # comupute the obj_val
    first_term1 = np.dot((label_matrix).flatten(),(np.log(net_out)).flatten().T)
    first_term2 = np.dot((np.ones(label_matrix.shape)-label_matrix).flatten(),(np.log(np.ones(net_out.shape)-net_out).flatten().T))
    
    #first_term = np.dot((label_matrix).flatten(),(np.log(net_out)).flatten().T)+np.dot(np.array(1-label_matrix).flatten(),(np.log(np.array(1-net_out)).flatten().T)
    #s
    first_term = first_term1+first_term2
    second_term = lambdaval*(np.dot(w1.flatten(),w1.flatten().T)+np.dot(w2.flatten(),w2.flatten().T))
    obj_val = -1/(training_data.shape[0])*first_term + second_term / (2*training_data.shape[0]) # finish off eqn (15)



    # Error function and Backpropagation
    #delta_l = np.array(net_out)*np.array(1-net_out)*np.array(label_matrix - net_out) # correspondes to eqn(9)
    delta_l = net_out-label_matrix
    #dev_lj = -1*np.dot(delta_l.T, hiden_out_bias) # correspondes to eqn(8)
    dev_lj = np.dot(delta_l.T, hiden_out_bias)
    grad_w2 = (dev_lj + lambdaval *w2)/ training_data.shape[0] #correspondes to eqn(16)
    w2_noBias = w2[:,0:-1]
    delta_j = np.array(hiden_out)*np.array(1-hiden_out) # correspondes to -(1-Zj)Zj in eqn(12)
    dev_ji = np.dot((np.array(delta_j)*np.array(np.dot(delta_l,w2_noBias))).T,training_data_bias) # correspondes to eqn(12)
    grad_w1 = (dev_ji+lambdaval*w1)/training_data.shape[0] #correnspondes to eqn(17)


    # Reshape the gradient matrices to a 1D array.
    grad_w1_reshape = np.ndarray.flatten(grad_w1.reshape((grad_w1.shape[0]*grad_w1.shape[1],1)))
    grad_w2_reshape = grad_w2.flatten()
    obj_grad_temp = np.concatenate((grad_w1_reshape.flatten(), grad_w2_reshape.flatten()),0)
    obj_grad = np.ndarray.flatten(obj_grad_temp)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    # Your code here
    input_bias = np.ones((data.shape[0],1))  # create a bias
    data_bias = np.concatenate((data, input_bias), axis=1)  # add bias to training data
    hiden_out = sigmoid(np.dot(data_bias, w1.T))  # 3.32 equtions 1 and 2
    hiden_bias = np.ones((1,hiden_out.T.shape[1]))  # create a bias
    hiden_out_bias = np.concatenate((hiden_out.T, hiden_bias), axis=0).T  # add bias to hidden_out data
    net_out = sigmoid(np.dot(hiden_out_bias,w2.T))  # 3.32 eqution 3 and 4, feed forward is complete.
    # Make a 1D vector of the predictions.
    return net_out.argmax(axis=1)

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels.T
    train_y = np.zeros(shape=(21100, 2))
    train_l = labels[0:21100]
    valid_y = np.zeros(shape=(2665, 2))
    valid_l = labels[21100:23765]
    test_y = np.zeros(shape=(2642, 2))
    test_l = labels[23765:]
    for i in range(train_y.shape[0]):
        train_y[i, train_l[i]] = 1
    for i in range(valid_y.shape[0]):
        valid_y[i, valid_l[i]] = 1
    for i in range(test_y.shape[0]):
        test_y[i, test_l[i]] = 1

    return train_x, train_y, valid_x, valid_y, test_x, test_y
"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
train_label = train_label[:,1]
validation_label=validation_label[:,1]
test_label=test_label[:,1]
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
