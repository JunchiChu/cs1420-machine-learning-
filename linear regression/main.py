#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the main program to read data, run classifiers
   and print results to stdout.

   You do not need to change this file. You can add debugging code or code to
   help produce your report, but this code should not be run by default in
   your final submission.

   Brown CS142, Spring 2020
"""

import numpy as np
import random
import gzip
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import LinearRegression

WINE_FILE_PATH = 'wine.txt'
#WINE_FILE_PATH = '../junch/ml/hw2/wine.txt'

def import_wine(filepath, test_size=0.2):
    '''
        Helper function to import the wine dataset

        @param:
            filepath: path to wine.txt
            test_size: the fraction of the dataset set aside for testing
        @return:
            X_train: training data inputs
            Y_train: training data values
            X_test: testing data inputs
            Y_test: testing data values
    '''

    # Check if the file exists
    if not os.path.exists(filepath):
        print('The file {} does not exist'.format(filepath))
        exit()

    # Load in the dataset
    data = np.loadtxt(filepath, skiprows=1)
    X, Y = data[:, 1:], data[:, 0]

    # Normalize the inputs
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def test_linreg():
    '''
        Helper function that tests LinearRegression.

        @param:
            None
        @return:
            None
    '''
    m = np.array([[2, 3], [1, 0]])
    mm = np.array([[2, 3,4,5], [1,1,3, 0]])
    for l in range(2):
        print(mm[l,range(2)])
    ###print(m)
    n=np.append(m, np.ones((len(m), 1)), axis=1)
    #print(n)
    #print(m.shape[1])
    s= LinearRegression(m.shape[1])
    #print(s.weights)


    X_train, X_test, Y_train, Y_test = import_wine(WINE_FILE_PATH)
    #print(X_train.shape[1])
    num_features = X_train.shape[1]

    # Padding the inputs with a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)
    #print(6.7**2)
    #### Matrix Inversion ######
    print('---- LINEAR REGRESSION w/ Matrix Inversion ---')
    solver_model = LinearRegression(num_features)
    solver_model.train(X_train_b, Y_train)
    print('Average Training Loss:', solver_model.average_loss(X_train_b, Y_train))
    print('Average Testing Loss:', solver_model.average_loss(X_test_b, Y_test))


def main():

    # Set random seeds. DO NOT CHANGE THIS IN YOUR FINAL SUBMISSION.
    random.seed(0)
    np.random.seed(0)
    test_linreg()

if __name__ == "__main__":
    main()
