import numpy as np
from metrics import accuracy_fn, mse_fn, macrof1_fn

def splitting_fn(data, labels, indices, fold_size, fold):
    """
        Function to split the data into training and validation folds.
        Arguments:
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            indices: (np.array, of shape (N,)): array of pre shuffled indices (integers ranging from 0 to N)
            fold_size (int): the size of each fold
            fold (int): the index of the current fold.
        Returns:
            train_data, train_label, val_data, val_label (np. arrays): split training and validation sets
    """
                
   
    #find the indices of the test samples among the training samples
    indices_val = indices[fold_size*fold : fold_size*fold + fold_size]
    #find the indices of the training samples among the training data
    indices_train = indices[[x not in indices_val for x in indices]] 
    
    val_data = data[indices_val]
    val_label = labels[indices_val]
    train_data = data[indices_train]
    train_label = labels[indices_train]
    return train_data, train_label, val_data, val_label

def cross_validation(method_obj=None, search_arg_name=None, search_arg_vals=[], data=None, labels=None, k_fold=4):
    """
        Function to run cross validation on a specified method, across specified arguments.
        Arguments:
            method_obj (object): A classifier or regressor object, such as KNN. Needs to have
                the functions: set_arguments, fit, predict.
            search_arg_name (str): the argument we are trying to find the optimal value for
                for example, for DummyClassifier, this is "dummy_arg".
            search_arg_vals (list): the different argument values to try, in a list.
                example: for the "DummyClassifier", the search_arg_name is "dummy_arg"
                and the values we try could be [1,2,3]
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            k_fold (int): number of folds
        Returns:
            best_hyperparam (float): best hyper-parameter value, as found by cross-validation
            best_acc (float): best metric, reached using best_hyperparam
    """
    ## choose the metric and operation to find best params based on the metric depending upon the
    ## kind of task.
    metric = mse_fn if method_obj.task_kind == 'regression' else macrof1_fn
    find_param_ops = np.argmin if method_obj.task_kind == 'regression' else np.argmax

    N = data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    fold_size = N//k_fold

    acc_list1 = []
    for arg in search_arg_vals:
        arg_dict = {search_arg_name: arg}
        # this is just a way of giving an argument 
        # (example: for DummyClassifier, this is "dummy_arg":1)
        method_obj.set_arguments(**arg_dict)

        acc_list2 = []
        for fold in range(k_fold):
            
            #separate our training samples into training and test samples
            train_data, train_label, val_data, val_label = splitting_fn(data,labels,indices,fold_size,fold)
            #use our splitting to initialize our classifier/regressor
            method_obj.fit(train_data,train_label)
            #add to our accuracy list the accuracy of our prediction
            acc_list2.append(metric(method_obj.predict(val_data),val_label))
        
         
        #compute the mean of all the accuracies for one parameter in order to compare each accuracy of each parameters
        acc_list1.append(np.mean(acc_list2))
     
    #find the best param in the given list search_arg_vals
    best_hyperparam = search_arg_vals[find_param_ops(acc_list1)]
    
    #find the accuracy for our best_hyperparam
    best_acc = acc_list1[(search_arg_vals).index(best_hyperparam)]


    return best_hyperparam, best_acc

        


    