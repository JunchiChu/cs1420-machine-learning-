#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Logistic Regression classifier

   Brown CS142, Spring 2020
'''
import random
import numpy as np

import math

def sig(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
    '''
    Apply softmax to an array

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

class LogisticRegression:
    '''
    Multinomial Logistic Regression that learns weights using 
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_features + 1, n_classes))  # An extra row added for the bias
        self.alpha = 0.03  # tune this parameter
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model, using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        # TODO
        epoch=0
        stop=1
        Zip=np.array([])
        for i in range(len(Y)):
               m=np.append([X[i]],[Y[i]])
               Zip=np.append(Zip,m)
        

        Zip=Zip.reshape(X.shape[0],X.shape[1]+1)

        useful_size=int(X.shape[0]/self.batch_size)*self.batch_size
        while(stop==1):
            np.random.shuffle(Zip)
            #print(Zip)
            batch_X=Zip[0:X.shape[0],0:X.shape[1]]
            batch_Y=Zip[0:X.shape[0],X.shape[1]]
            # print("******")
            # print(batch_X)
            # print(batch_Y)
            #print("******")
            pre_loss=self.loss(X,Y)
            for k in range(0,useful_size,self.batch_size):
               #print(k)

               
               l=np.dot(batch_X[k:k+self.batch_size],self.weights)
               p = l
               for m in range(p.shape[0]):
                    p[m]=softmax(p[m])
               #print(p.shape)
               for s in range(0,0+self.batch_size):
                   p[s][int(batch_Y[s+k])]=p[s][int(batch_Y[s+k])]-1
               #print(p)

               Lw=np.dot(np.transpose(batch_X[k:k+self.batch_size]),p)
               #pre_loss=self.loss(X,Y)
               self.weights=self.weights-(0.03*Lw)/self.batch_size
            current_loss=self.loss(X,Y)
            #print(current_loss)
            if float(abs(current_loss-pre_loss))<self.conv_threshold :
                print("finish traning")
                return epoch 
            epoch=epoch+1
            #print(current_loss)
    
            
        return epoch     

       
                


        


    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of datapoints.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the squared error of the model on the dataset
        '''
        # TODO
       
        trip_y = np.dot(X,self.weights)
        for x in range(trip_y.shape[0]):
            trip_y[x]=softmax(trip_y[x])
        sum=0
        for i in range(len(Y)):
               if trip_y[i][int(Y[i])]!=0:
                 sum=sum+np.log(trip_y[i][int(Y[i])])
        return -sum/X.shape[0]


    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        # TODO
        trip_y = np.dot(X,self.weights)
        for x in range(trip_y.shape[0]):
            trip_y[x]=softmax(trip_y[x])
        sing_y = np.ones(X.shape[0])
        for i in range(trip_y.shape[0]):
            sing_y[i] = np.where(trip_y[i]==max(trip_y[i]))[0][0]
        #sing_y = np.array([])
        # for i in range(trip_y.shape[0]):
        #     if  max(trip_y[i][0],trip_y[i][1],trip_y[i][2])==trip_y[i][0]:
        #         value=0
        #     elif max(trip_y[i][0],trip_y[i][1],trip_y[i][2])==trip_y[i][1]:
        #         value=1
        #     elif max(trip_y[i][0],trip_y[i][1],trip_y[i][2])==trip_y[i][2]:
        #         value=2
        #     sing_y = np.append(sing_y, [value], axis=0)
        #print(sing_y)
        return sing_y.reshape(len(sing_y),)

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        # TODO
        trip_y = (np.dot(X,self.weights))
        for x in range(trip_y.shape[0]):
            trip_y[x]=softmax(trip_y[x])
        sing_y = np.ones(X.shape[0])
        for i in range(trip_y.shape[0]):
            sing_y[i] = np.where(trip_y[i]==max(trip_y[i]))[0][0]
            
        # for i in range(trip_y.shape[0]):
        #     if  max(trip_y[i][0],trip_y[i][1],trip_y[i][2])==trip_y[i][0]:
        #         value=0
        #     elif max(trip_y[i][0],trip_y[i][1],trip_y[i][2])==trip_y[i][1]:
        #         value=1
        #     elif max(trip_y[i][0],trip_y[i][1],trip_y[i][2])==trip_y[i][2]:
        #         value=2
        #     sing_y = np.append(sing_y, [value], axis=0)

        #sing_y=self.predict(X)

        match = 0
        for x in range(sing_y.shape[0]):
            if sing_y[x]==Y[x]:
                match = match+1
        return match/sing_y.shape[0]
  
        
