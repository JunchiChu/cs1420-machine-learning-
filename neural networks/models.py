import numpy as np
import random


def l2_loss(predictions,Y):
    '''
        Computes L2 loss (sum squared loss) between true values, Y, and predictions.

        :param Y: A 1D Numpy array with real values (float64)
        :param predictions: A 1D Numpy array of the same size of Y
        :return: L2 loss using predictions for Y.
    '''
    # TODO
    dif=np.sum((Y-predictions)**2)
    return dif

def sigmoid(x):
    '''
        Sigmoid function f(x) =  1/(1 + exp(-x))
        :param x: A scalar or Numpy array
        :return: Sigmoid function evaluated at x (applied element-wise if it is an array)
    '''
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))
def relu(x):
    return np.where(x > 0, x, 0)

def relu_deri(x):
    return np.where(x > 0, 1, 0)


def sigmoid_derivative(x):
    '''
        First derivative of the sigmoid function with respect to x.
        :param x: A scalar or Numpy array
        :return: Derivative of sigmoid evaluated at x (applied element-wise if it is an array)
    '''
    # TODO
    return sigmoid(x)*(1-sigmoid(x))

class OneLayerNN:
    '''
        One layer neural network trained with Stocastic Gradient Descent (SGD)
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the neural network model.
        '''
        self.weights = None
        pass

    def train(self, X, Y, learning_rate=0.001, epochs=250, print_loss=True):
        '''
        Trains the OneLayerNN model using SGD.

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :param learning_rate: The learning rate to use for SGD
        :param epochs: The number of times to pass through the dataset
        :param print_loss: If True, print the loss after each epoch.
        :return: None
        '''
        # TODO
        self.weights=np.random.normal(0,0.1,(X.shape[1],))
        Zip=np.array([])
        for i in range(len(Y)):
          m=np.append([X[i]],[Y[i]])
          Zip=np.append(Zip,m)
        Zip=Zip.reshape(X.shape[0],X.shape[1]+1)
        

        #print(Y.shape)
        for x in range(epochs):
          np.random.shuffle(Zip)
          SX=Zip[:,0:X.shape[1]]
          SY=Zip[:,X.shape[1]]
          #print(self.average_loss(X,Y))
          for y in range(SX.shape[0]):
            deri=2*(SX[y]@self.weights-SY[y])*SX[y]
            self.weights=self.weights-learning_rate*deri
        

        

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X: 2D Numpy array where each row contains an example.
        :return: A 1D Numpy array containing the corresponding predicted values for each example
        '''
        # TODO
        return X@self.weights

        

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]


class TwoLayerNN:

    def __init__(self, hidden_size, activation=sigmoid, activation_derivative=sigmoid_derivative):
        '''
        @attrs:
            activation: the activation function applied after the first layer
            activation_derivative: the derivative of the activation function. Used for training.
            hidden_size: The hidden size of the network (an integer)
            output_neurons: The number of outputs of the network
        '''
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size

        # In this assignment, we will only use output_neurons = 1.
        self.output_neurons = 1

    def _get_layer2_bias_gradient(self, x, y, layer1_weights, layer1_bias,
        layer2_weights, layer2_bias):
        '''
        Computes the gradient of the loss with respect to the output bias, b2.

        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :param layer1_weights: Numpy array of dimension: hidden_size by input_size
        :param layer1_bias: Numpy array of dimension: hidden_size by 1
        :param layer2_weights: Numpy array of dimension: output_neurons by hidden_size
        :param layer2_bias: Numpy array of dimension: output_neurons by 1
        :return: the partial derivates dL/db2, a numpy array of dimension: output_neurons by 1
        '''
        # TODO
        hx=layer2_weights@(self.activation(layer1_weights@x+layer1_bias))+layer2_bias
        return (2*hx-2*y)

    def _get_layer2_weights_gradient(self, x, y, layer1_weights, layer1_bias,
        layer2_weights, layer2_bias):
        '''
        Computes the gradient of the loss with respect to the output weights, v.

        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :param layer1_weights: Numpy array of dimension: hidden_size by input_size
        :param layer1_bias: Numpy array of dimension: hidden_size by 1
        :param layer2_weights: Numpy array of dimension: output_neurons by hidden_size
        :param layer2_bias: Numpy array of dimension: output_neurons by 1
        :return: the partial derivates dL/dv, a numpy array of dimension: output_neurons by hidden_size
        '''
        # TODO
        lay1_out=self.activation(layer1_weights@x+layer1_bias)
        hx=layer2_weights@lay1_out+layer2_bias
        #print("hx",hx.shape)
        #print("lay1_out".shape,lay1_out.shape)
        return (2*hx-2*y)*np.transpose(lay1_out)
       


    def _get_layer1_bias_gradient(self, x, y, layer1_weights, layer1_bias,
        layer2_weights, layer2_bias):
        '''
        Computes the gradient of the loss with respect to the hidden bias, b1.

        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :param layer1_weights: Numpy array of dimension: hidden_size by input_size
        :param layer1_bias: Numpy array of dimension: hidden_size by 1
        :param layer2_weights: Numpy array of dimension: output_neurons by hidden_size
        :param layer2_bias: Numpy array of dimension: output_neurons by 1
        :return: the partial derivates dL/db1, a numpy array of dimension: hidden_size by 1
        '''
        # TODO
        lay1_out=self.activation(layer1_weights@x+layer1_bias)
        hx=layer2_weights@lay1_out+layer2_bias
        return (2*(hx-y))*layer2_weights.T*self.activation_derivative(layer1_weights@x+layer1_bias)
    

    def _get_layer1_weights_gradient(self, x, y, layer1_weights, layer1_bias,
        layer2_weights, layer2_bias):
        '''
        Computes the gradient of the loss with respect to the hidden weights, W.

        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :param layer1_weights: Numpy array of dimension: hidden_size by input_size
        :param layer1_bias: Numpy array of dimension: hidden_size by 1
        :param layer2_weights: Numpy array of dimension: output_neurons by hidden_size
        :param layer2_bias: Numpy array of dimension: output_neurons by 1
        :return: the partial derivates dL/dW, a numpy array of dimension: hidden_size by input_size
        '''
        # TODO
        lay1_out=self.activation(layer1_weights@x+layer1_bias)
        hx=layer2_weights@lay1_out+layer2_bias

        # print("qqqqq")
        # print("x",x.shape)
        # print("sd",(layer1_weights@x).shape)
        # print("layer1 weight",layer1_weights.shape)
        # print("layer1 bias",layer1_bias.shape)
        # print("layer1 out",lay1_out.shape)
        # print("hx-y",(2*(hx-y)).shape)
        # print("layer2 weights",layer2_weights.shape)
        # print((2*(hx-y)*layer2_weights.T).shape)
        # print((self.activation_derivative(lay1_out)).shape)

        result=(2*(hx-y))*layer2_weights.T*self.activation_derivative(layer1_weights@x+layer1_bias)*np.transpose(x)
        #print("result",result.shape)
        return (2*(hx-y))*layer2_weights.T*self.activation_derivative(layer1_weights@x+layer1_bias)*np.transpose(x)

    def train(self, X, Y, learning_rate=0.01, epochs=2000, print_loss=True):
        '''
        Trains the TwoLayerNN with SGD using Backpropagation.

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :param learning_rate: The learning rate to use for SGD
        :param epochs: The number of times to pass through the dataset
        :param print_loss: If True, print the loss after each epoch.
        :return: None
        '''
        # NOTE:
        # Use numpy arrays of the following dimensions for your model's parameters.
        # layer 1 weights: hidden_size x input_size
        # layer 1 bias: hidden_size x 1
        # layer 2 weights: output_neurons x hidden_size
        # layer 2 bias: output_neurons x 1
        # HINT: for best performance initialize weights with np.random.normal or np.random.uniform
        # TODO
        self.yshape=X.shape[0]
        # self.layer1_weights=np.zeros((self.hidden_size,X.shape[1]))
        # self.layer1_bias=np.zeros((self.hidden_size,1))
        # self.layer2_weights=np.zeros((self.output_neurons,self.hidden_size))
        # self.layer2_bias=np.zeros((self.output_neurons,1))
        self.layer1_weights=np.random.normal(0,0.1,(self.hidden_size,X.shape[1]))
        self.layer1_bias=np.random.normal(0,0.1,(self.hidden_size,1))
        self.layer2_weights=np.random.normal(0,0.1,(self.output_neurons,self.hidden_size))
        self.layer2_bias=np.random.normal(0,0.1,(self.output_neurons,1))
        print("out_new",self.output_neurons)
        Zip=np.array([])
        for i in range(len(Y)):
          m=np.append([X[i]],[Y[i]])
          Zip=np.append(Zip,m)
        Zip=Zip.reshape(X.shape[0],X.shape[1]+1)
        

        #print(Y.shape)
        for x in range(epochs):
          np.random.shuffle(Zip)
          SX=Zip[:,0:X.shape[1]]
          SY=Zip[:,X.shape[1]]
          #print(self.loss(X,Y))
          for y in range(SX.shape[0]):
            ix=SX[y].reshape((SX[y].shape[0],1))
            #print(SY[y])
            iy=SY[y].reshape((1,1))
            lsw=(self._get_layer1_bias_gradient(ix,iy,self.layer1_weights,self.layer1_bias,self.layer2_weights,self.layer2_bias),
            self._get_layer1_weights_gradient(ix,iy,self.layer1_weights,self.layer1_bias,self.layer2_weights,self.layer2_bias),
            self._get_layer2_bias_gradient(ix,iy,self.layer1_weights,self.layer1_bias,self.layer2_weights,self.layer2_bias),
            self._get_layer2_weights_gradient(ix,iy,self.layer1_weights,self.layer1_bias,self.layer2_weights,self.layer2_bias))
            #print(lsw[0].shape)
            self.layer1_bias-=learning_rate*lsw[0]
            self.layer1_weights-=learning_rate*lsw[1]
            self.layer2_bias-=learning_rate*lsw[2]
            self.layer2_weights-=learning_rate*lsw[3]
        # print("l1b",self.layer1_bias.shape,"l1w",self.layer1_weights.shape)
        # print("l2b",self.layer2_bias.shape,"l2w",self.layer2_weights.shape)
        

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X: 2D Numpy array where each row contains an example.
        :return: A 1D Numpy array containing the corresponding predicted values for each example
        '''
        # TODO
        # print("l1weight",self.layer1_weights.shape)
        # print("X.shape",X.shape)
        # print("l1 b",self.layer1_bias.shape)
        #layer1_output=self.activation(self.layer1_weights@X+self.layer1_bias)
        result=np.ones((self.yshape,))
        for ind in range(X.shape[0]):
            ix=X[ind].reshape((X[ind].shape[0],1))
            layer1_output=self.activation(self.layer1_weights@ix+self.layer1_bias)
            layer2_output=self.layer2_weights@layer1_output+self.layer2_bias
            result[ind]=layer2_output
        
        return result
        

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the squared error of the model on the dataset
        '''
        self.yshape=X.shape[0]
        predictions = self.predict(X)
      
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]
