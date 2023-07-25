import numpy as np

class PCA(object):
    """
        PCA dimensionality reduction object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, find_principal_components, and reduce_dimension work correctly.
    """
    def __init__(self, *args, **kwargs):
        """
            You don't need to initialize the task kind for PCA.
            Call set_arguments function of this class.
        """
        self.set_arguments(*args, **kwargs)
        #the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        #the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The PCA class should have a variable defining the number of dimensions (d).
            You can either pass this as an arg or a kwarg.
        """
        
        if 'd' in kwargs:
            self.d = kwargs["d"]
        elif len(args) > 0:
            self.d = args[0]
        else:
            self.d = 30

    def find_principal_components(self, training_data):
        """
            Finds the principal components of the training data. Returns the explained variance in percentage.
            IMPORTANT: 
            This function should save the mean of the training data and the principal components as
            self.mean and self.W, respectively.

            Arguments:
                training_data (np.array): training data of shape (N,D)
            Returns:
                exvar (float): explained variance
        """
        
        #Computing the mean of the training data and saving it
        self.mean = np.mean(training_data, 0)
        
        #Center the training data with mean
        training_data_tilde = training_data - self.mean
        
        #Covariance Matrix
        covMatrix = (training_data_tilde.T@training_data_tilde) / training_data_tilde.shape[0]
        
        #Computing eigenvectors and eigenvalues
        eigvals, eigvecs = np.linalg.eigh(covMatrix)
        
        #Sorting eigenvalues and eigenvector by decreasing order  
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        #Take d eigenvectors to compute the W matrix (and saving it) and the explained variance
        self.W = eigvecs[:, 0 : self.d]
        eg = eigvals[0 : self.d]
        
        exvar = 100*eg.sum()/eigvals.sum()
            
        return exvar

    def reduce_dimension(self, data):
        """
            Reduce the dimensions of the data, using the previously computed
            self.mean and self.W. 

            Arguments:
                data (np.array): data of shape (N,D)
            Returns:
                data_reduced (float): reduced data of shape (N,d)
        """
        data_reduced = (data - self.mean) @self.W
        return data_reduced
        

