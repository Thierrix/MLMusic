import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import accuracy_fn, macrof1_fn
import matplotlib.pyplot as plt
## MS2!!


class SimpleNetwork(nn.Module):
    """
    A network which does classification!
    """
    def __init__(self, input_size, num_classes, hidden_size=32):
        super(SimpleNetwork, self).__init__()
        #initialize activation function tanh
        self.tanh = nn.Tanh()

        #initilaize a dropout function
        self.dropout = nn.Dropout(p=0.6)
       
        #initilaize the layers of our model
        self.fc1 = nn.Linear(input_size, input_size//3)
        self.fc2 = nn.Linear(input_size//3, input_size//2) 
        self.fc3 = nn.Linear(input_size//2, num_classes) 
        
    def forward(self, x):
        """
        Takes as input the data x and outputs the 
        classification outputs.
        Args: 
            x (torch.tensor): shape (N, D)
        Returns:
            output_class (torch.tensor): shape (N, C) (logits)
        """
        

        #output_class = self.leakyrelu(self.fc1(x))
       
        #first layer with an activation function
        output_class = self.tanh(self.fc1(x))

        #dropout on the output class in order to prevent overfitting
        output_class = self.dropout(output_class)

        #second layer with an activation function
        output_class = self.tanh(self.fc2(output_class))
        
        #last layer
        output_class = self.fc3(output_class)

        return output_class

class Trainer(object):

    """
        Trainer class for the deep network.
    """

    def __init__(self, model, lr, epochs, beta=100):
        """
        """
        self.lr = lr
        self.epochs = epochs
        self.model= model
        self.beta = beta
        self.accuracy = 0 #variable used to register the accuracy of our model

        self.classification_criterion = nn.CrossEntropyLoss()
        #set the optimizer to AdamW optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        #function use in order to do the classification
        self.f = torch.nn.Softmax()
    def train_all(self, dataloader_train, dataloader_val):
        """
        Method to iterate over the epochs. In each epoch, it should call the functions
        "train_one_epoch" (using dataloader_train) and "eval" (using dataloader_val).
        """

        #set several variables, in order to do early stopping to prevent overfitting
        patience = 7
        no_improvement_epochs = 0
        best_val_loss = float('-inf')

        for ep in range(self.epochs):
            self.train_one_epoch(dataloader_train)
            self.eval(dataloader_val)
            val_loss = self.accuracy
            
            if (ep+1) % 50 == 0: #reduce the learning rate each 25 epochs
                print("Reduce Learning rate")

                for g in self.optimizer.param_groups:
                    g["lr"] = g["lr"]*0.9 #decrement the learning rate of 0.8

            #If the validation loss is the best seen so far, update the best loss and reset the counter
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
            #Otherwise, increment the counter
            else:
                no_improvement_epochs += 1
            
            # If the counter reaches the patience, stop training
            if no_improvement_epochs == patience:
                print("Early stopping at epoch {}".format(ep))
                break

    def train_one_epoch(self, dataloader):
        """
        Method to train for ONE epoch.
        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode!
        i.e. self.model.train()
        """
        # Training.
        self.model.train()
        for it,batch in enumerate(dataloader):
            #Load a batch, break it down in images and targets.
            x, _, y = batch
           
            #Run forward pass.
            logits = self.model(x)
            
            #Compute loss (using 'criterion').
            loss = self.classification_criterion(logits, y)

            #Run backward pass.
            loss.backward()
            
            #Update the weights using optimizer.
            self.optimizer.step()
            
            #Zero-out the accumulated gradients.
            self.optimizer.zero_grad()
            
            
        
    def eval(self, dataloader):
        """
            Method to evaluate model using the validation dataset OR the test dataset.
            Don't forget to set your model to eval mode!
            i.e. self.model.eval()
            Returns:
                Note: N is the amount of validation/test data. 
                We return one torch tensor which we will use to save our results (for the competition!)
                results_class (torch.tensor): classification results of shape (N,)
        """
        #set the model to eval
        self.model.eval()

        #initialize result_class to it's final shape  
        N = len(dataloader.dataset)         
        result_class = torch.randn(N,)
        
       
        #parameter used for slicing 
        counterBatch = 0
        with torch.no_grad():
            accuracy_run = 0
            for it, batch in enumerate(dataloader):
                # Get batch of data
                x,_,y = batch
                #size of the batch
                bs = x.shape[0]
                
                #slice result class to give the correct prediction to the correct place in result_class
                #in order to do classification, we apply the softmax function on the model and we then take the argmax of it to resolve to 1 class y
                result_class[counterBatch:(counterBatch + bs)] = torch.argmax(self.f(self.model(x)), dim = 1) 

                #calculate the accuracy for each batch
                accuracy_run += (torch.mean((y ==  result_class[counterBatch:(counterBatch + bs)]), dtype = float))*bs

                #increment the parameter used for slicing 
                counterBatch += bs
                
            acc = accuracy_run/len(dataloader.dataset)

        #register the accuracy
        print("The accuracy is " ,acc.numpy()*100)
        self.accuracy = acc*100 

        return result_class


#I found this activation function on github, so tried it, https://github.com/pytorch/pytorch/issues/76799
@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))