"""
    This is the class file you will have to fill in using helper functions defined in kmeans.py.
"""
import numpy as np

from kmeans import kmeans
import scipy
from scipy.spatial import distance

class KmeansClassifier(object):
    """
    K-Means Classifier via Iterative Improvement
    @attrs:
        k: The number of clusters to form as well as the number of centroids to
           generate (default = 10), an int
        tol: Value specifying our convergence criterion. If the ratio of the
             distance each centroid moves to the previous position of the centroid
             is less than this value, then we declare convergence.
        max_iter: the maximum number of times the algorithm can iterate trying
                  to optimize the centroid values, an int,
                  the default value is set to 500 iterations
        cluster_centers_: a Numpy array where each element is one of the k cluster centers
    """

    def __init__(self, n_clusters = 10, max_iter = 500, threshold = 1e-6):
        """
        Initiate K-Means with some parameters
        """
        self.k = n_clusters
        self.tol = threshold
        self.max_iter = max_iter
        self.cluster_centers_ = np.array([])

    def train(self, X):
        """
        Compute K-Means clustering on each class label and store your result in self.cluster_centers_
        :param X: inputs of training data, a 2D Numpy array
        :return: None
        """
        # TODO (hint: use kmeans())

        self.cluster_centers_=kmeans(X,self.k,self.max_iter,self.tol)
    

    def predict(self, X, centroid_assignments):
        """
        Predicts the label of each sample in X based on the assigned centroid_assignments.

        :param X: A dataset as a 2D Numpy array
        :param centroid_assignments: a Numpy array of 10 digits (0-9) representing the interpretations of the digits of the plotted centroids
        :return: A Numpy array of predicted labels
        """

        # TODO: complete this step only after having plotted the centroids!
        euclidean_dist=distance.cdist(X,self.cluster_centers_,'euclidean')
        return np.array(centroid_assignments)[np.argmin(euclidean_dist,axis=1)]
    
        

    def accuracy(self, data, centroid_assignments):
        """
        Compute accuracy of the model when applied to data
        :param data: a namedtuple including inputs and labels
        :return: a float number indicating accuracy
        """
        pred = self.predict(data.inputs, centroid_assignments)
        return np.mean(pred == data.labels)
