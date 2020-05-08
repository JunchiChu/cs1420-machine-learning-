import numpy as np
import sys
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import OneLayerNN, TwoLayerNN


def test_models(dataset, epochs, test_size=0.2):
    '''
        Tests LinearRegression, OneLayerNN, TwoLayerNN on a given dataset.
        :param dataset The path to the dataset
        :return None
    '''

    # Check if the file exists
    if not os.path.exists(dataset):
        print('The file {} does not exist'.format(dataset))
        exit()

    # Load in the dataset
    data = np.loadtxt(dataset, skiprows = 1)
    X, Y = data[:, 1:], data[:, 0]

    # Normalize the features
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    print('Running models on {} dataset'.format(dataset))

    # Add a bias
    #print("dsdsds")
    print(X_train.shape)
    #print(Y_train.shape)
    #print(Y[0:100])
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    #### 1-Layer NN ######
    print('----- 1-Layer NN -----')
    nnmodel = OneLayerNN()
    nnmodel.train(X_train_b, Y_train, epochs=epochs, print_loss=False)
    print('Average Training Loss:', nnmodel.average_loss(X_train_b, Y_train))
    print('Average Testing Loss:', nnmodel.average_loss(X_test_b, Y_test))

    #### 2-Layer NN ######
    print('----- 2-Layer NN -----')
    model = TwoLayerNN(10)
    # Use X without a bias, since we learn a bias in the 2 layer NN.
    model.train(X_train, Y_train, epochs=epochs, print_loss=False)
    print('Average Training Loss:', model.average_loss(X_train, Y_train))
    print('Average Testing Loss:', model.average_loss(X_test, Y_test))

def test_gradients():
    '''
    Tests the gradient functions of TwoLayerNN.

    :return: nothing
    '''
    model = TwoLayerNN(2)

    # Fake training example
    x = np.array([[15.0], [-5.5]])
    y = np.array([2.0])

    # Initial model parameters
    layer1_weights = np.array(
        [[0.05796310320578392, -0.060187073024398074],
        [0.10586356541552687, -0.24996779540007905]])
    layer1_bias = np.array([[0.06711270248424306],[0.10896375475941099]])
    layer2_weights = np.array([[0.5248657232614117, 0.7456542888037705]])
    layer2_bias = np.array([[1.19300293108928]])

    # Expected gradients
    expected_layer1_weights = np.array(
        [[0.8507615520850692, -0.3119459024311921],
        [0.2984432314293994, -0.10942918485744645]])
    expected_layer1_bias = np.array([[0.05671743680567129],[0.019896215428626626]])
    expected_layer2_weights = np.array([[0.491922676291208, 0.6024843192405677]])
    expected_layer2_bias = np.array([[0.6304037119950578]])

    print('----Testing 2-Layer NN Gradients-----')

     # Test layer 1 weights
    print("\nTesting layer one weights gradient.")
    actual_layer1_weights = model._get_layer1_weights_gradient(x, y, layer1_weights, layer1_bias, layer2_weights, layer2_bias)
    if not hasattr(actual_layer1_weights, "shape"):
        print("Layer one weights gradient is not a numpy array.")
    elif actual_layer1_weights.shape != expected_layer1_weights.shape:
        print(
            "Incorrect shape for layer one weights gradient.\nExpected: {0}\nActual: {1}".format(
            expected_layer1_weights.shape, actual_layer1_weights.shape))
    elif not np.all(np.isclose(actual_layer1_weights, expected_layer1_weights)):
        print(
            "Incorrect values for layer one weights gradient.\nExpected: {0}\nActual: {1}".format(
            expected_layer1_weights, actual_layer1_weights))
    else:
        print("Layer one weights gradient is correct.")

    # Test layer 1 bias
    print("\nTesting layer one bias gradient.")
    actual_layer1_bias = model._get_layer1_bias_gradient(x, y, layer1_weights, layer1_bias, layer2_weights, layer2_bias)
    if not hasattr(actual_layer1_bias, "shape"):
        print("Layer one bias gradient is not a numpy array.")
    elif actual_layer1_bias.shape != expected_layer1_bias.shape:
        print(
            "Incorrect shape for layer one bias gradient.\nExpected: {0}\nActual: {1}".format(
            expected_layer1_bias.shape, actual_layer1_bias.shape))
    elif not np.all(np.isclose(actual_layer1_bias, expected_layer1_bias)):
        print(
            "Incorrect values for layer one bias gradient.\nExpected: {0}\nActual: {1}".format(
            expected_layer1_bias, actual_layer1_bias))
    else:
        print("Layer one bias gradient is correct.")

    # Test layer 2 weights
    print("\nTesting layer two weights gradient.")
    actual_layer2_weights = model._get_layer2_weights_gradient(x, y, layer1_weights, layer1_bias, layer2_weights, layer2_bias)
    if not hasattr(actual_layer2_weights, "shape"):
        print("Layer two weights gradient is not a numpy array.")
    elif actual_layer2_weights.shape != expected_layer2_weights.shape:
        print(
            "Incorrect shape for layer two weights gradient.\nExpected: {0}\nActual: {1}".format(
            expected_layer2_weights.shape, actual_layer2_weights.shape))
    elif not np.all(np.isclose(actual_layer2_weights, expected_layer2_weights)):
        print(
            "Incorrect values for layer two weights gradient.\nExpected: {0}\nActual: {1}".format(
            expected_layer2_weights, actual_layer2_weights))
    else:
        print("Layer two weights gradient is correct.")

    # Test layer 2 bias
    print("\nTesting layer two bias gradient.")
    actual_layer2_bias = model._get_layer2_bias_gradient(x, y, layer1_weights, layer1_bias, layer2_weights, layer2_bias)
    if not hasattr(actual_layer2_bias, "shape"):
        print("Layer two bias gradient is not a numpy array.\n")
    elif actual_layer2_bias.shape != expected_layer2_bias.shape:
        print(
            "Incorrect shape for layer two bias gradient.\nExpected: {0}\nActual: {1}\n".format(
            expected_layer2_bias.shape, actual_layer2_bias.shape))
    elif not np.all(np.isclose(actual_layer2_bias, expected_layer2_bias)):
        print(
            "Incorrect values for layer two bias gradient.\nExpected: {0}\nActual: {1}\n".format(
            expected_layer2_bias, actual_layer2_bias))
    else:
        print("Layer two bias gradient is correct.\n")

def main():
    # Set random seeds. DO NOT CHANGE THIS IN YOUR FINAL SUBMISSION.
    random.seed(0)
    np.random.seed(0)
    # Uncomment to test gradient calculating functions for 2-layer NN
    test_gradients()
    test_models('data/wine.txt', 25)
    print("random")


if __name__ == "__main__":
    main()
