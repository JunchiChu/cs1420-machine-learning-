"""
    This is a file you will have to fill in.
    It contains helper functions required by K-means method via iterative improvement
"""
import numpy as np
from random import sample
from scipy.spatial import distance
from numpy import linalg as LA

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    Hint: use random.sample (it is already imported for you!)
    :param k: number of cluster centroids
    :param inputs: a 2D Numpy array, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    centroids=np.ones((k,inputs.shape[1]))
    all_pts=[]
    for i in range(0, inputs.shape[0]): 
     all_pts.append(i) 
    result=sample(all_pts,k)
    for s in range(k):
        centroids[s]=inputs[result[s]]
    return centroids


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    :param inputs: inputs of data, a 2D Numpy array
    :param centroids: a Numpy array of k current centroids
    :return: a Numpy array of centroid indices, one for each row of the inputs
    """
    # TODO
    result=np.ones((inputs.shape[0]),dtype = int)
    for x in range(inputs.shape[0]):
        dis_for_each_c=[]
        for y in range(centroids.shape[0]):
         #print((centroids[y]))
         dis_for_each_c.append(distance.euclidean(inputs[x], centroids[y]))
        #print(centroids[int(dis_for_each_c.index(min(dis_for_each_c)))])
        result[x]=dis_for_each_c.index(min(dis_for_each_c))
    #print(result[0:50])
    return result



def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    :param inputs: inputs of data, a 2D Numpy array
    :param indices: a Numpy array of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    #print(type(indices[5]))
    new_centroids=np.zeros((k,inputs.shape[1]))
    count=np.zeros((k,inputs.shape[1]))
    # old_centroid=np.unique(indices).tolist()

    for x in range(inputs.shape[0]):
        new_centroids[indices[x]]+=inputs[x]
        count[indices[x]]+=1
    for y in range(k):
        new_centroids[y]/=count[y]
    #print(new_centroids.shape)
    #print("new cen",new_centroids)
    return new_centroids


    


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    :param inputs: inputs of data, a 2D Numpy array
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: relative tolerance with regards to inertia to declare convergence, a float number
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    new_centeroids=init_centroids(k,inputs)
    count_iter=0
    count_tol=0
    stop_sign=0
    while(count_iter<=max_iter and stop_sign!=k):
        stop_sign=0   
        old_cen=new_centeroids
        index=assign_step(inputs,new_centeroids)
        new_centeroids=update_step(inputs,index,k)
        for x in range(k):
           # print(k)
            if (LA.norm(old_cen[x]-new_centeroids[x])/LA.norm(old_cen[x])) <tol:
                stop_sign+=1
        count_iter+=1
        #print(stop_sign)
    if count_iter==max_iter:
        print("touch max iter")
    #print(new_centeroids)
    return new_centeroids

