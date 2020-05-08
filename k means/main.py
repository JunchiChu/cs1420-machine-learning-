"""
    This file contains the main program to read data, run classifier and print results.
    To run the main.py file from command line, simply navigate to the directory where main.py resides, and type:
        python main.py
"""
import sys
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import namedtuple
import random
from sklearn.model_selection import train_test_split

from models import KmeansClassifier

## KMEANS HELPERS ##

def plot_Kmeans(model):
    """
        Takes in a pre-trained K-Means classifier model and plots the 10 centroids.
        Note: this function is designed only for the digits.csv data set.
    :param model: pre-trained K-Means classifier model object
    :return: None
    """
    if isinstance(model, KmeansClassifier) == False:
        print("Invalid input! Model must be a KmeansClassifier object.")
        return

    cluster_centers = model.cluster_centers_
    #print(cluster_centers)
    #cluster_centers=[1,2,3,4,5,6,7,8,9,0]
    fig, ax = plt.subplots(1, len(cluster_centers), figsize=(3, 1))
    
    for i in range(len(cluster_centers)):
        axi = ax[i]
        center = cluster_centers[i]
        #print(center.shape)
        center = np.array(center).reshape(8,8)
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    plt.show()

def test_Kmeans(model, test_data, centroid_assignments):
    """
        Prints the accuracy of model on test_data, based on the centroid ordering provided by the student.
    :param model: pre-trained K-Means classifier model object
    :param test_data: a namedtuple including test inputs and test train_labels
    :param centroid_assignments: a python list of 10 digits (0-9) representing your interpretations of the digits of the plotted centroids from plot_Kmeans (in order from left ot right).
    :return: None
    """
    if isinstance(centroid_assignments, list) == False:
        print("Invalid input! Centroid assignments must be a python list!")
        return
    # elif not np.array_equal(np.array(list(range(10))), np.sort(np.array(centroid_assignments))):
    #     print("Invalid Input! Centroid assignments must contain all numbers in the range 0-9 (in the order displayed in your plot).")
    #     return 
    elif isinstance(model, KmeansClassifier) == False:
        print("Invalid input! Model must be a KmeansClassifier object.")
        return
   
    accuracy = model.accuracy(test_data, centroid_assignments)
    print("Testing on K-Means Classifier (K = " + str(model.k) + "), the accuracy is {:.2f}%".format(accuracy * 100))

def runKMeans():
    '''
        Trains, plots, and tests K-Means classifier on digits.csv dataset.
    '''
    NUM_CLUSTERS = 10# DO NOT CHANGE
    random.seed(1) # DO NOT CHANGE
    np.random.seed(1) # DO NOT CHANGE

    Dataset = namedtuple('Dataset', ['inputs', 'labels'])

    # Read data
    data = pd.read_csv("data/digits.csv", header = 0)

    # We assume labels are in the first column of the dataset
    labels = data.values[:, 0]

    # If labels are of type string, convert class names to numeric values
    if isinstance(labels[0], str):
        classes = np.unique(labels)
        class_mapping = dict(zip(classes, range(0, len(classes))))
        labels = np.vectorize(class_mapping.get)(labels)

    # Features columns are indexed from 1 to the end, make sure that dtype = float32
    inputs = data.values[:, 1:].astype("float32")

    # Split data into training set and test set with a ratio of 2:1
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size = 0.33)

    all_data = Dataset(inputs, labels)
    train_data = Dataset(train_inputs, train_labels)
    test_data = Dataset(test_inputs, test_labels)
    print("Shape of training data inputs: ", train_data.inputs.shape)
    print("Shape of test data inputs:", test_data.inputs.shape)

    # Train K-Means Classifier
    kmeans_model = KmeansClassifier(NUM_CLUSTERS)
    kmeans_model.train(train_data.inputs)
    

    # DO NOT MODIFY ABOVE THIS LINE!

    # TODO: uncomment below to plot the centroids for the 10 digits (0-9).
    plot_Kmeans(kmeans_model)

    # TODO: fill out centroid_assignments below based on the visualization of plot_Kmeans (in order from left to right). In this step, you are assigning each centroid to its most resembling digit (0-9).
    test_Kmeans(kmeans_model, test_data, centroid_assignments=[9,2,1,4,3,0,8,6,5,7])
    #9,2,1,4,3,0,8,6,5,7
    #16-8,3,2,4,2,9,1,6,1,7,5,5,4,0,2,3
    #4=9,1,7,6

# DO NOT MODIFY BELOW
def main():
    runKMeans()

if __name__ == '__main__':
    main()