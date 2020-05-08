import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

class RegularizedLogisticRegression(object):
    '''
    Implement regularized logistic regression for binary classification.

    The weight vector w should be learned by minimizing the regularized risk
    log(1 + exp(-y <w, x>)) + \lambda \|w\|_2^2. In other words, the objective
    function is the log loss for binary logistic regression plus Tikhonov
    regularization with a coefficient of \lambda.
    '''
    def __init__(self):
        self.learningRate = 0.00001 # Feel free to play around with this if you'd like, though this value will do
        self.num_epochs = 10000 # Feel free to play around with this if you'd like, though this value will do
        self.batch_size = 15 # Feel free to play around with this if you'd like, though this value will do
        self.weights = None

        #####################################################################
        #                                                                    #
        #    MAKE SURE TO SET THIS TO THE OPTIMAL LAMBDA BEFORE SUBMITTING    #
        #                                                                    #
        #####################################################################

        self.lmbda = 1 # tune this parameter

    def train(self, X, Y):
        '''
        Train the model, using batch stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None
        '''
        #[TODO]
        lmmda=self.lmbda
        tem_epo=self.num_epochs
        self.weights=np.zeros((X[0].shape[0] , 1)) 
        #print(self.weights)
        Zip=np.array([])
        for i in range(len(Y)):
               m=np.append([X[i]],[Y[i]])
               Zip=np.append(Zip,m)
        Zip=Zip.reshape(X.shape[0],X.shape[1]+1)
        useful_size=int(X.shape[0]/self.batch_size)*self.batch_size
        b=10
        while(tem_epo!=0):
            np.random.shuffle(Zip)
            shuffled_X=Zip[0:X.shape[0],0:X.shape[1]]
            shuffled_Y=Zip[0:X.shape[0],X.shape[1]:X.shape[1]+1]
            # l=np.dot(shuffled_X[0:0+self.batch_size],self.weights)
            # p=l-(shuffled_Y[0:0+self.batch_size].reshape(-1,1))
            # Lw=(np.dot(np.transpose(shuffled_X[0:0+self.batch_size]),p)/self.batch_size) +2*lmmda*(self.weights)
            for k in range(0,useful_size,self.batch_size):
               l=np.dot(shuffled_X[k:k+self.batch_size],self.weights)
               l=sigmoid_function(l)
               p=l-(shuffled_Y[k:k+self.batch_size].reshape(-1,1))
               Lw=(np.dot(np.transpose(shuffled_X[k:k+self.batch_size]),p)/self.batch_size) +2*lmmda*(self.weights)
               self.weights=self.weights-(self.learningRate*Lw)
               #print(self.weights)

            tem_epo=tem_epo-1
        #print(self.weights)
        
    
            
             



    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        #[TODO]
        #print(self.weights)
        after_sig = sigmoid_function(np.dot(X,self.weights))
        for i in range(after_sig.shape[0]):
            if after_sig[i]>=0.5:
                after_sig[i]=1
            else:
                after_sig[i]=0
        #print(after_sig)
        return after_sig

    def accuracy(self,X, Y):
        '''
        Output the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        #[TODO]
        pred=self.predict(X)
        #print(Y)
        match=0
        for i in range(pred.shape[0]):
            if pred[i]==Y[i]:
                match=match+1
        return match/pred.shape[0]
          

    def runTrainTestValSplit(self, lambda_list, X_train, Y_train, X_val, Y_val):
        '''
        Given the training and validation data, fit the model with training data and test it with
        respect to each lambda. Record the training error and validation error.
        @params:
            lambda_list: a list of lambdas
            X_train: a 2D Numpy array for trainig where each row contains an example,
            padded by 1 column for the bias
            Y_train: a 1D Numpy array for training containing the corresponding labels for each example
            X_val: a 2D Numpy array for validation where each row contains an example,
            padded by 1 column for the bias
            Y_val: a 1D Numpy array for validation containing the corresponding labels for each example
        @returns:
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
        '''
        train_errors = np.ones(len(lambda_list))
        val_errors = np.ones(len(lambda_list))
        count=0
        #[TODO] train model and calculate train and validation errors here for each lambda
        for l in lambda_list:
            self.lmbda=l
            self.train(X_train,Y_train)
            train_errors[count]=1-self.accuracy(X_train,Y_train)
            val_errors[count]=1-self.accuracy(X_val,Y_val)
            count=count+1
        return train_errors, val_errors

    def _kFoldSplitIndices(self, dataset, k):
        '''
        Helper function for k-fold cross validation. Evenly split the indices of a
        dataset into k groups.

        For example, indices = [0, 1, 2, 3] with k = 2 may have an output
        indices_split = [[1, 3], [2, 0]].

        Please don't change this.
        @params:
            dataset: a Numpy array where each row contains an example
            k: an integer, which is the number of folds
        @return:
            indices_split: a list containing k groups of indices
        '''
        num_data = dataset.shape[0]
        fold_size = int(num_data / k)
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        indices_split = np.split(indices[:fold_size*k], k)
        return indices_split

    def runKFold(self, lambda_list, X, Y, k = 3):
        '''
        Run k-fold cross validation on X and Y with respect to each lambda. Return all k-fold
        errors.

        Each run of k-fold involves k iterations. For an arbitrary iteration i, the i-th fold is
        used as testing data while the rest k-1 folds are training data. The k results are
        averaged as the cross validation error.
        @params:
            lambda_list: a list of lambdas
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
            k: an integer, which is the number of folds, k is 3 by default
        @return:
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        '''
        k_fold_errors = np.ones(len(lambda_list))
        err=0
        count=0
        bag_ind=np.arange(k)
        for lmbda in lambda_list:
            self.lmbda = lmbda
            #[TODO] call _kFoldSplitIndices to split indices into k groups randomly
            split_result=self._kFoldSplitIndices(np.arange(X.shape[0]),k)
            

            #[TODO] for each iteration i = 1...k, train the model using lmbda
            # on k−1 folds of data. Then test with the i-th fold.
            for x in range(k):
                fold_x_test=np.array([])
                fold_y_test=np.array([])
                fold_x_train=np.array([])
                fold_y_train=np.array([])
                k_minusone=np.delete(bag_ind,x)
                indices_test=split_result[x]
                indices_train=np.array([])
                for ele in k_minusone:
                    indices_train=np.append(indices_train,split_result[ele])
                

                for test_e in indices_test:
                    fold_x_test=np.append(fold_x_test,X[test_e])
                    fold_y_test=np.append(fold_y_test,Y[test_e])
                fold_x_test=fold_x_test.reshape(-1,X.shape[1])
                fold_y_test=fold_y_test.reshape(-1,1)


                for train_e in indices_train:
                    #print(train_e)
                    fold_x_train=np.append(fold_x_train,X[int(train_e)])
                    fold_y_train=np.append(fold_y_train,Y[int(train_e)])
                fold_x_train=fold_x_train.reshape(-1,X.shape[1])
                fold_y_train=fold_y_train.reshape(-1,1)
                
                self.train(fold_x_train,fold_y_train)
                tem_err=1-self.accuracy(fold_x_test,fold_y_test)#calcul error 
                err = err+tem_err

            k_fold_errors[count]=err/k
            err=0
            count=count+1


            #[TODO] calculate and record the cross validation error by averaging total errors

        return k_fold_errors

    def plotError(self, lambda_list, train_errors, val_errors, k_fold_errors):
        '''
        Produce a plot of the cost function on the training and validation sets, and the
        cost function of k-fold with respect to the regularization parameter lambda. Use this plot
        to determine a valid lambda.
        @params:
            lambda_list: a list of lambdas
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        @return:
            None
        '''
        plt.figure()
        plt.semilogx(lambda_list, train_errors, label = 'training error')
        plt.semilogx(lambda_list, val_errors, label = 'validation error')
        plt.semilogx(lambda_list, k_fold_errors, label = 'k-fold error')
        plt.xlabel('lambda')
        plt.ylabel('error')
        plt.legend()
        plt.show()
