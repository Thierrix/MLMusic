import numpy as np

class KNN(object):
    """
        kNN classifier object.
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
            The KNN class should have a variable defining the number of neighbours (k).
            You can either pass this as an arg or a kwarg.
        """
        if 'k' in kwargs:
            self.knn_neighbours = kwargs["k"]
        elif len(args) > 0:
            self.knn_neighbours = args[0]
        else:
            self.knn_neighbours = 5
    
    def euclidean_distances(self, tr_ex, ex):
        """
            Computes the euclidean distance between the tr_exs and x.
            Arguments:
                ex (np.array) : example of shape (D,)
                tr_ex (np.array) : other training examples of shape (N, D).
            Output :
                returns distance vector of length N
        """
        res = ((tr_ex - ex)**2).sum(axis = 1)
        return np.sqrt(res)
    
    def get_k_neighbors(self, k, distances):
        """
            Gets a number of k neigbours according to the smallest distance
            Arguments:
                k (int) : the number of neighbours we want to get
                distances (np.array): array of all distances
            Output:
                indices (np.array) : array of size k gathering the k smallest distances 
        """
        indices = np.argsort(distances)[:k]
        return indices
    
    def predict_label(self, neighbors_labels):
        """
            Gets the most predicted label between the neighbors
            Arguments:
                neighbor_labels (np.array) : the labels the neighbors have
            Output:
                The most occurring label between all the neighbors
        """
        return np.argmax(np.bincount(neighbors_labels))
    
    def knn_one_example(self, unlabeled_example, training_features, training_labels, k):
        """
           Computes the k-NearestNeighbors technique on one example
           Arguments:
                unlabeled_example (np.array) : example of shape (D,) 
                training_features : training features of the data
                training_labels : training labels of the data 
                k (int) : the number of neighbours we want to get
            Output : 
                The best label we can get for the unlabeled example.
        """
        #Get the euclidean distances between the example and the training_data
        dist = self.euclidean_distances(unlabeled_example, training_features)
        
        #Get the k-Nearest Neighbors then get their labels
        nearest_neighbors_ind = self.get_k_neighbors(k, dist)
        neighbors_labels = training_labels[nearest_neighbors_ind]
        
        #Get the max-occuring label
        best_label = self.predict_label(neighbors_labels)
        
        return best_label
            
    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.
            
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        
        self.training_data = training_data
        self.training_labels = training_labels
        
        pred_labels = np.apply_along_axis(func1d = self.knn_one_example, 
                                          axis = 1, 
                                          arr=training_data, 
                                          training_features = training_data,
                                          training_labels = training_labels,
                                          k = self.knn_neighbours)
        return pred_labels
                               
    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """    

        test_labels = np.apply_along_axis(func1d = self.knn_one_example,
                                          axis = 1,
                                          arr = test_data,
                                          training_features = self.training_data,
                                          training_labels = self.training_labels,
                                          k = self.knn_neighbours)
        return test_labels