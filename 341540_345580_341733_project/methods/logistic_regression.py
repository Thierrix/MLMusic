import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot


class LogisticRegression(object):
    """
        LogisticRegression classifier object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """

    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """
        
       
        if 'lr' in kwargs:
            self.lr = kwargs['lr']
        elif len(args) > 0 :
            self.lr = args[0]
            
        if 'max_iters' in kwargs:
            self.max_iters = kwargs['max_iters']
        elif len(args) > 0 :
            self.max_iters = args[1]
    
    def f_softmax(self, X,w):
        """
            Computes the softmax function relatively to the training data and the weights w.
            Arguments :
                X (np.array) : training data of shape (N, D)
                w (np.array) : Weights of shape (D, C) where C is the number of classes
                Returns :
                res (np.array) :  Probabilities of shape (N, C) where each value is in range[0, 1]
                                  and each row sums to 1.
        """
        #compute the probability for all our samples
        proba = np.exp(X@w)
        #sum all probabilities for each class C
        S = np.sum(proba, axis = 1, keepdims = True)
        #normalize our data in order to have the sum of a label according to C sum up to 1 
        res = np.divide(proba, S)
        return res
    
    def gradient_logistic(self, X, y, w):
        """
            Arguments :
                X (np.array): Input data of shape (N, D)
                y (np.array): Labels of shape  (N,C)
                w (np.array): Weights of shape (D, C)
            Returns :
                grad (np.array): Gradients of shape (D, C)
        """
        res = self.f_softmax(X, w)
        #our y is og shape(N,) and we need to have y in shape (N,C) so we need to do a one hot encoding
        y_onehot = label_to_onehot(y)
        #calculate the gradient
        grad = X.T@(res-y_onehot)
        return grad
        
    def logistic_regression_train(self, X,y):
        """
            Classification function train using logistic regression.
            Arguments :
                X (np.array): Data of shape (N, D).
                y (np.array): Labels of shape (N, C)
            Returns:
                w : np. array: Label assignments of data of shape (N,C).
        """
        #initialize our weights
        w = np.random.normal(0.1,0.2,[X.shape[1],self.C])
        
        #calculate the best w by doing gradient descent max_iters time
        for it in range(self.max_iters):
            w = w - self.lr * self.gradient_logistic(X,y,w)
             
        return w 
    
    def logistic_regression_classify(self, X,w):
        """
            Classification function for multi class logistic regression. 
    
            Args:
                X (np.array): Data of shape (N, D).
                w (np.array): Weights of logistic regression model of shape (D, C)
            Returns:
                predictions (np.array): Label assignments of data of shape (N,)
        """
        #select the class C where we have the higher probability to assign it to this class
        predictions = 1*np.array(np.argmax(self.f_softmax(X,w),axis = 1))
        return predictions
    
    
    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,)
            Returns:
                pred_labels (np.array): target of shape (N,)
        """
        #intitializwe parameters D and C (it is clearer like this)
        self.D, self.C = training_data.shape[1], int(np.max(training_labels)+1)
        #find the best w according to our training samples
        self.w = self.logistic_regression_train(training_data,training_labels)
        #compute the prediction on the test data
        pred_labels = self.logistic_regression_classify(training_data, self.w)
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.s
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """   
        #predict the tet_labels value 
        pred_labels = self.logistic_regression_classify(test_data,self.w)
        return pred_labels