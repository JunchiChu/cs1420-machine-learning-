import numpy as np

class NaiveBayes(object):
    """ Bernoulli Naive Bayes model
    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes model with n_classes. """
        self.n_classes = n_classes
        self.att_distri=np.ones((2,69))
        self.pri_distri=np.ones((2,1))

    def train(self, X_train, y_train):
        """ Trains the model, using maximum likelihood estimation.
        @params:
            X_train: a n_examples x n_attributes numpy array
            y_train: a n_examples numpy array
        @return:
            a tuple consisting of:
                1) a 2D numpy array of the attribute distributions
                2) a 1D numpy array of the priors distribution
        """

        # TODO
        #print(self.n_classes)
        
        p_yes=np.count_nonzero(y_train == 1)/y_train.size
        p_no=np.count_nonzero(y_train == 0)/y_train.size
        att_distri=np.ones((4,X_train.shape[1]))
        self.att_distri=att_distri
        pri_distri=np.ones((2,1))
        pri_distri[0][0]=p_yes
        pri_distri[1][0]=p_no
        for x in range(X_train.shape[1]):
            given_yes_label_yes=given_yes_label_no=given_no_label_yes=given_no_label_no=0
            
            for y in range(y_train.size):
                #print(X_train[y][x])
                if y_train[y] ==1:
                    if X_train[y][x]==1:
                        given_yes_label_yes+=1
                    else:
                        given_yes_label_no+=1

                if y_train[y] ==0:
                    if X_train[y][x]==1:
                        given_no_label_yes+=1
                    else:
                        given_no_label_no+=1

            att_distri[0][x]=(given_yes_label_yes+1)/(np.count_nonzero(y_train == 1)+2)
            att_distri[1][x]=(given_yes_label_no+1)/(np.count_nonzero(y_train == 1)+2) 
            att_distri[2][x]=(given_no_label_yes+1)/(np.count_nonzero(y_train == 0)+2)
            att_distri[3][x]=(given_no_label_no+1)/(np.count_nonzero(y_train == 0)+2) 
            #print(att_distri[0][x]+att_distri[1][x]+att_distri[3][x]+att_distri[2][x])
            # if (att_distri[0][x]+att_distri[1][x]+att_distri[3][x]+att_distri[2][x])!=2:
            #     print("wrong")
        self.att_distri=att_distri
        self.pri_distri=pri_distri
        #print(self.att_distri)
        #print(self.pri_distri)
        return (att_distri,pri_distri)

        


    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """

        # TODO
        result=np.ones((inputs.shape[0],1))
        for d in range(inputs.shape[0]):
           pro_yes_input=pro_no_input=1
           for m in range(inputs.shape[1]):
                if inputs[d][m]==1:
                 pro_yes_input*=self.att_distri[0][m]
                 pro_no_input*=self.att_distri[2][m]
                else:
                 pro_yes_input*=self.att_distri[1][m]
                 pro_no_input*=self.att_distri[3][m]
           pro_yes_input*=self.pri_distri[0]
           pro_no_input*=self.pri_distri[1]  
           norm_yes=pro_yes_input/(pro_yes_input+pro_no_input)
           norm_no=pro_no_input/(pro_yes_input+pro_no_input)
           if norm_yes>norm_no:
                result[d]=1
           else:
                result[d]=0
        return result


        
    def accuracy(self, X_test, y_test):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            X_test: 2D numpy array of examples
            y_test: numpy array of labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """

        # TODO
        prediction=self.predict(X_test)
        correct=0
        for y in range(prediction.size):
            if prediction[y]==y_test[y]:
                correct+=1
        #print(correct/y_test.size)
        return correct/y_test.size



    def print_fairness(self, X_test, y_test, x_sens):
        """ 
        ***DO NOT CHANGE what we have implemented here.***
        
        Prints measures of the trained model's fairness on a given dataset (data).

        For all of these measures, x_sens == 1 corresponds to the "privileged"
        class, and x_sens == 1 corresponds to the "disadvantaged" class. Remember that
        y == 1 corresponds to "good" credit. 

        @params:
            X_test: 2D numpy array of examples
            y_test: numpy array of labels
            x_sens: numpy array of sensitive attribute values
        @return:

        """
        predictions = self.predict(X_test)
        predictions=predictions.flatten()
        # Disparate Impact (80% rule): A measure based on base rates: one of
        # two tests used in legal literature. All unprivileged lasses are
        # grouped together as values of 0 and all scp s are given
        # the class 1. . Given data set D = (X,Y, C), with protected
        # attribute X (e.g., race, sex, religion, etc.), remaining attributes Y,
        # and binary class to be predicted C (e.g., “will hire”), we will say
        # that D has disparate impact if:
        # P[Y^ = 1 | S != 1] / P[Y^ = 1 | S = 1] <= (t = 0.8). 
        # Note that this 80% rule is based on US legal precedent; mathematically,
        # perfect "equality" would mean

        di = np.mean(predictions[np.where(x_sens==0)])/np.mean(predictions[np.where(x_sens==1)])
        print("Disparate impact: " + str(di))

        # Group-conditioned error rates! False positives/negatives conditioned on group
        
        pred_priv = predictions[np.where(x_sens==1)]
        pred_unpr = predictions[np.where(x_sens==0)]
        y_priv = y_test[np.where(x_sens==1)]
        y_unpr = y_test[np.where(x_sens==0)]

        # s-TPR (true positive rate) = P[Y^=1|Y=1,S=s]
        priv_tpr = np.sum(np.logical_and(pred_priv == 1, y_priv == 1))/np.sum(y_priv)
        unpr_tpr = np.sum(np.logical_and(pred_unpr == 1, y_unpr == 1))/np.sum(y_unpr)

        # s-TNR (true negative rate) = P[Y^=0|Y=0,S=s]
        priv_tnr = np.sum(np.logical_and(pred_priv == 0, y_priv == 0))/(len(y_priv) - np.sum(y_priv))
        unpr_tnr = np.sum(np.logical_and(pred_unpr == 0, y_unpr == 0))/(len(y_unpr) - np.sum(y_unpr))

        # s-FPR (false positive rate) = P[Y^=1|Y=0,S=s]
        priv_fpr = 1 - priv_tnr 
        unpr_fpr = 1 - unpr_tnr 

        # s-FNR (false negative rate) = P[Y^=0|Y=1,S=s]
        priv_fnr = 1 - priv_tpr 
        unpr_fnr = 1 - unpr_tpr

        print("FPR (priv, unpriv): " + str(priv_fpr) + ", " + str(unpr_fpr))
        print("FNR (priv, unpriv): " + str(priv_fnr) + ", " + str(unpr_fnr))
    
    
        # #### ADDITIONAL MEASURES IF YOU'RE CURIOUS #####

        # Calders and Verwer (CV) : Similar comparison as disparate impact, but
        # considers difference instead of ratio. Historically, this measure is
        # used in the UK to evalutate for gender discrimination. Uses a similar
        # binary grouping strategy. Requiring CV = 1 is also called demographic
        # parity.

        cv = 1 - (np.mean(predictions[np.where(x_sens==1)]) - np.mean(predictions[np.where(x_sens==0)]))

        # Group Conditioned Accuracy: s-Accuracy = P[Y^=y|Y=y,S=s]

        priv_accuracy = np.mean(predictions[np.where(x_sens==1)] == y_test[np.where(x_sens==1)])
        unpriv_accuracy = np.mean(predictions[np.where(x_sens==0)] == y_test[np.where(x_sens==0)])

        return predictions
