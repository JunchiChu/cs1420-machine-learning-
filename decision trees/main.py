import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from models import DecisionTree, train_error, entropy, gini_index

def mylossgraph(ax, title, tree_array, train_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    for x in range(15):
       ax.plot(tree_array[x].loss_plot_vec(train_data), label='train non-pruned with depth'+str(x+1))

    # ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    # ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    # ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def explore_dataset(filename, class_name):




    my_train_data=np.array([[1,0,0,0,1],[0,0,0,1,1],[1,1,1,1,0],[1,0,0,1,1],[0,0,0,0,1],[1,1,0,0,0]])
    #print(my_train_data)
    # play_tree=DecisionTree(data=my_train_data,gain_function=train_error)
    # play_tree1=DecisionTree(data=my_train_data,gain_function=gini_index)
    # play_tree.loss(my_train_data)
    # play_tree1.loss(my_train_data)
    #2 largest gain 
    #print(play_tree._calc_gain(my_train_data,2,play_tree.gain_function))
    # print(play_tree.root.left.isleaf)
    # print(play_tree1.root.left.isleaf)



    train_data, validation_data, test_data = get_data(filename, class_name)
    #print(train_data.shape)



    # print("start printing TRAIN data loss without pruning...order is:entropy/train error/gini")
    # mytree_entropy_x=DecisionTree(data=train_data,validation_data=None,gain_function=entropy)
    # mytree_trainerror_x=DecisionTree(data=train_data,validation_data=None,gain_function=train_error)
    # mytree_gini_x=DecisionTree(data=train_data,validation_data=None,gain_function=gini_index)
    # print(mytree_entropy_x.loss(train_data))
    # print(mytree_trainerror_x.loss(train_data))
    # print(mytree_gini_x.loss(train_data))
    # print("end of printing TRAIN data loss WITHOUT pruning...order is:entropy/train error/gini")
    # print("******************************************************************************")
    # print("start printing TEST data loss WITHOUT pruning...order is:entropy/train error/gini")
    # print(mytree_entropy_x.loss(test_data))
    # print(mytree_trainerror_x.loss(test_data))
    # print(mytree_gini_x.loss(test_data))
    # print("end of printing TEST data loss WITHOUT pruning...order is:entropy/train error/gini")
    # print("********************************************************************")


    # mytree_entropy=DecisionTree(data=train_data,validation_data=validation_data,gain_function=entropy)
    # mytree_trainerror=DecisionTree(data=train_data,validation_data=validation_data,gain_function=train_error)
    # mytree_gini=DecisionTree(data=train_data,validation_data=validation_data,gain_function=gini_index)
    # print("start printing TRAIN data loss WITH pruning...order is:entropy/train error/gini")
    # print(mytree_entropy.loss(train_data))
    # print(mytree_trainerror.loss(train_data))
    # print(mytree_gini.loss(train_data))
    # print("end of printing TRAIN data loss WITH pruning...order is:entropy/train error/gini")
    # print("******************************************************************************")
    # print("start printing TEST data loss WITH pruning...order is:entropy/train error/gini")
    # print(mytree_entropy.loss(test_data))
    # print(mytree_trainerror.loss(test_data))
    # print(mytree_gini.loss(test_data))
    # print("end of printing TEST data loss WITH pruning...order is:entropy/train error/gini")
    # print("********************************************************************")
    # print(mytree_entropy.root.right.right.index_split_on)
    ax=plt.axes()
 


    # loss_plot(ax,"train_error",mytree_trainerror_x,mytree_trainerror,train_data,test_data)
    # plt.show()


    # loss_plot(ax,"entropy",mytree_entropy_x,mytree_entropy,train_data,test_data)
    # plt.show()



    # loss_plot(ax,"gini",mytree_gini_x,mytree_gini,train_data,test_data)
    # plt.show()
    ax=plt.axes()

    # for x in range(1,16):
    #     mytree=DecisionTree(data=train_data,validation_data=None,gain_function=entropy,max_depth=x)
    #     print("setup:using entropy function---our max depth is",x,"---our loss is",mytree.loss(train_data))
    #     loss_plot(ax,"loss",mytree,mytree,train_data,test_data)
    #     plt.show()
    tree=[]
    depth=[]
    for x in range(1,16):
       mytree=DecisionTree(data=train_data,validation_data=None,gain_function=entropy,max_depth=x)
       tree.append(mytree.loss(train_data))
       depth.append(x)
    #print(len(tree))
    #mylossgraph(ax,"graph",tree,train_data)
    #plt.show()
    

    plt.plot(depth,tree,'ro')
    plt.title("relation between depth and loss")
    plt.xlabel('depth info')
    plt.ylabel('loss info without pruning--train data set ')
    plt.show()


    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    #      (b) Print average test loss (not-pruned)
    #      (c) Print average training loss (pruned)
    #      (d) Print average test loss (pruned)

    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!

def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################
    # print("Let's printing chess data set...CHESSSSSSSSSSSSS")
    # explore_dataset('data/chess.csv', 'won')
    # print("End of printing chess data set")
    # print("##############################")
    # print("##############################")
    # print("##############################")
    # print("##############################")
    # print("##############################")
    # print("##############################")
    # print("##############################")
    # print("Let's printing spam data set...SPAMMMMMMMMMMMMMMM")
    explore_dataset('data/spam.csv', '1')
    print("End of printing spam data set")
    

main()
