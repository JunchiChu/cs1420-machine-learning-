import numpy as np
import random
import copy
import math

def train_error(prob):
    '''
        TODO:
        Calculate the train error of the subdataset and return it.
        For a dataset with two classes, C(p) = min{p, 1-p}
    '''
    return min(prob,1-prob)


def entropy(prob):
    '''
        TODO:
        Calculate the entropy of the subdataset and return it.
        For a dataset with 2 classes, C(p) = -p * log(p) - (1-p) * log(1-p)
        For the purposes of this calculation, assume 0*log0 = 0.
    '''
    if prob==0:
       return_value=0
    elif prob==1:
        return_value=0
    else:
        return_value=-prob*(np.log(prob))-(1-prob)*(np.log(1-prob))

   

    return return_value


def gini_index(prob):
    '''
        TODO:
        Calculate the gini index of the subdataset and return it.
        For dataset with 2 classes, C(p) = 2 * p * (1-p)
    '''
    return 2*prob*(1-prob)



class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} # used for visualization


    def _set_info(self, gain, num_samples):
        '''
        Helper function to add to info attribute.
        You do not need to modify this. 
        '''

        self.info['gain'] = gain
        self.info['num_samples'] = num_samples


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)


    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)


    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)


    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        #print(cnt/len(data))
        return cnt/len(data)


    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        '''
        if node==None:
            return 
        self._prune_recurs(node.left,validation_data)
        self._prune_recurs(node.right,validation_data)

        if node.isleaf==True:
             return
        if node.isleaf==False:
          if node.left.isleaf==False or node.right.isleaf==False:
              return 
        if node.left.isleaf==True and node.right.isleaf==True:
            prev_loss=self.loss(validation_data)
            tem_l=copy.deepcopy(node.left)
            tem_r=copy.deepcopy(node.right)
            node.left=None
            node.right=None
            node.isleaf=True
            curr_loss=self.loss(validation_data)
            if curr_loss>=prev_loss:
                node.left=tem_l
                node.right=tem_r
                node.isleaf=False
            return     
            


        
        
        


    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion if:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node reaches the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf.
            - A label, indicating the label of the leaf (or the label it would 
              be if we were to terminate at that node)
        '''
        

        if data.shape[0]!=0:
            major_vote=[]
            for x in range(data[:,0].shape[0]):
              major_vote.append(data[:,0][x])
            counts = np.bincount(major_vote)
            node.label=np.argmax(counts)


        if data.shape[0]==0 or len(indices)==0 or node.depth==self.max_depth:
            return (True,node.label)
    
        
       
        flag=False
        compare_label=data[0][0]
        for x in range(data[:,0].shape[0]):
            if data[:,0][x]!=compare_label:
                flag=True
        if flag==False:
            return (True,node.label)
        return (False,node.label)
         


    def _split_recurs(self, node, data, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.

        First use _is_terminal() to check if the node needs to be split.
        If so, select the column that has the maximum infomation gain to split on.
        Store the label predicted for this node, the split column, and use _set_info()
        to keep track of the gain and the number of datapoints at the split.
        Then, split the data based on its value in the selected column.
        The data should be recursively passed to the children.
        '''
        if self._is_terminal(node,data,indices)[0] is True:      
            node.isleaf=True
            return 
        else:
            major_vote=[]
            for x in range(data[:,0].shape[0]):
                major_vote.append(data[:,0][x])
            counts = np.bincount(major_vote)
            node.label=np.argmax(counts)
            node.isleaf=False
            list_right=[]
            list_left=[]

            best_index=np.ones(len(indices))
            for m in range(len(indices)): 
              best_index[m]=self._calc_gain(data,indices[m],self.gain_function)
            max_gain=np.amax(best_index)
            #print(max_gain)
            max_index=indices[np.where(best_index==max_gain)[0][0]]
            #max_index=indices[np.argmax(max_gain)]
            node.index_split_on=max_index
            #print(max_index)
            
            node._set_info(max_gain,data.shape[0])
            # data_left=np.ones((1,data.shape[1]),dtype = int)
            # data_right=np.ones((1,data.shape[1]),dtype = int)
            # for x in range(data[:,max_index].shape[0]):
            #     if data[:,max_index][x]==1:
            #        #print("hhohoohohoho")
            #        data_right=np.concatenate((data_right, np.array([data[x]])), axis=0)
            #     else:                 
            #        data_left=np.concatenate((data_left, np.array([data[x]])), axis=0)
            # data_left=np.delete(data_left, 0, 0)
            # data_right=np.delete(data_right, 0, 0)

            
            for x in range(data[:,max_index].shape[0]):
                if data[:,max_index][x]==1:
                    list_right.append(x)
                else:
                    list_left.append(x)
            data_left=np.ones([len(list_left),data.shape[1]])
            data_right=np.ones([len(list_right),data.shape[1]])
            indices.remove(max_index)
    
        
            if len(list_left)==0:
               data_left=np.array([])
            
            else:
              iter_cnt=0
              for x in np.nditer(np.transpose(list_left)):
                data_left[iter_cnt]=data[x]
                iter_cnt=iter_cnt+1
                

            if len(list_right)==0:
                data_right=np.array([])
              
            else:
               iter_cnt=0
               for y in np.nditer(np.transpose(list_right)):
                 data_right[iter_cnt]=data[y]
                 iter_cnt=iter_cnt+1
               iter_cnt=0
        
            
            
            left_node=Node(depth = node.depth+1,label=0)
            right_node=Node(depth = node.depth+1,label=1)
         
            node.left=left_node
            node.right=right_node

            right_indices=copy.deepcopy(indices)
            left_indices=copy.deepcopy(indices)
            self._split_recurs(left_node,data_left,left_indices)
            
            
            self._split_recurs(right_node,data_right,right_indices)


            


    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - (P[x_i=True] * C(P[y=1|x_i=True]) + (P[x_i=False] * C(P[y=1|x_i=False]))
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        
        count_xi_false=count_xi_true=count_y_1=condi_y1_xt=condi_y1_xf=0
        for x in range(data[:,split_index].shape[0]):
            if data[:,split_index][x] == 1:
                if data[:,0][x]==1:
                    condi_y1_xt=condi_y1_xt+1
                count_xi_true=count_xi_true+1
            else:
                if data[:,0][x]==1:
                    condi_y1_xf=condi_y1_xf+1
                count_xi_false=count_xi_false+1
        if count_xi_false!=0:
           prob_condi_y1_xf=condi_y1_xf/count_xi_false
        else:
           prob_condi_y1_xf=0

        if count_xi_true!=0:
          prob_condi_y1_xt=condi_y1_xt/count_xi_true
        else:
          prob_condi_y1_xt=0

        prob_xi_true=count_xi_true/(data[:,split_index].shape[0])
        prob_xi_false=1-prob_xi_true

        for x in range(data[:,0].shape[0]):
            if data[:,0][x]==1:
                count_y_1=count_y_1+1
        prob_y_1=count_y_1/(data[:,split_index].shape[0])

        gain_value=gain_function(prob_y_1)-prob_xi_true*gain_function(prob_condi_y1_xt)-prob_xi_false*gain_function(prob_condi_y1_xf)
        return gain_value


    

    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = {:d}; gain = {:f}; number of samples = {:d}'.format(node.index_split_on, node.info['gain'], node.info['num_samples'])
            left = indent + '0 -> '+ print_subtree(node.left, indent + ' ')
            right = indent + '1 -> '+ print_subtree(node.right, indent + ' ')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')


    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)


    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if not row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
