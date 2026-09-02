from utils.gradientDescent import SGD
import torch
import numpy as np
import matplotlib.pyplot as plt

class Classifier():
    def __init__(self, lambda_param : float):
        self.lambda_param = lambda_param
    
    def predict(self, x):
        if x.dtype in [np.float64, np.int64]:
            x = torch.from_numpy(x).float()

        output = torch.matmul(x, self.weight) + self.bias
        pred_label = torch.sign(output)
        return torch.where(pred_label <= -1, 0, 1)
    def loss_fn(self, data):
        x = data[0] 
        y = data[1]
        y_hat = self.predict(x)
        return (self.lambda_param * self.weight.norm()**2 + torch.mean(torch.clamp(1-y_hat*y, 0, 1)))
    
    def update_weights(self, x, y, lr):
        num_samples = x.shape[0]
        
        for i in range(num_samples):
            # Extract individual sample vector and scalar label
            x_i = x[i].reshape(1, -1) # Reshape to 2D matrix row (1, num_features)
            y_i = y[i]
            
            # Compute decision margin score for this sample
            y_hat_i = torch.matmul(x_i, self.weight) + self.bias
            condition = y_i * y_hat_i >= 1
            
            # Evaluate single boolean scalar value using .item()
            if condition.item():
                dw = 2 * self.lambda_param * self.weight
                db = 0
            else:
                # Transpose x_i to match the column shape of self.weight
                dw = 2 * self.lambda_param * self.weight - torch.matmul(x_i.t(), y_i.reshape(1, 1))
                db = y_i
                
            # Update weights incrementally per sample (Stochastic Gradient Descent)
            self.weight = self.weight - lr * dw
            self.bias = self.bias - lr * db

    def train(self, train_data, val_data, epochs: int, learning_rate = 0.001):
        '''
        Training function to train the model using provided the training and validation data
        
        Parameters
        ----------
        train_data : tuple
            A tuple containing the training data and labels (X_train, y_train)
        val_data : tuple
            A tuple containing the validation data and labels (X_val, y_val)
        epochs : int
            The number of epochs to train the model for
        '''

        self.m , self.n = train_data[0][0].shape
        self.weight = torch.normal(0, 0.01, (self.n , 1))
        self.bias = torch.zeros(1)
        self.lr = learning_rate
        
        batch_size = self.m
        self.loss_mean_store = []
        self.val_loss_mean_store = []

        total_loss = 0
        total_val_loss = 0
        ##ingesting the data, converting it back to x and y - but now in vector format
        num_bts = len(train_data)
        
        for epoch in range(epochs):
            loss_mean = 0
            print('-'*10 , epoch+1, '-'*10)
            for bts in range(num_bts):
                batch = train_data[bts]
                x = batch[0]
                y = batch[1]
                
                y_hat = self.predict(x)
                self.update_weights(x, y, self.lr)
                
                loss_mean += self.loss_fn(batch)
                
            self.loss_mean_store.append(loss_mean)
            loss_mean = loss_mean/num_bts
            total_loss += loss_mean
            val_loss_mean = self.evaluate(val_data)
            self.val_loss_mean_store.append(val_loss_mean)
            total_val_loss += val_loss_mean
            print(f'Mean Train Loss in Epoch: {epoch + 1} is: {loss_mean}')
            print(f'Mean Val Loss in Epoch: {epoch + 1} is: {val_loss_mean}')
            
        self.w_final = self.weight
        self.b_final = self.bias
        print('='*40)
        print('\n')
        print(f'final train loss: {loss_mean} \nfinal val loss: {val_loss_mean}')
        print(f'final weights: {self.w_final} \nfinal bias: {self.b_final}')
            
    
    def history(self,):
        epoch_train_loss = self.loss_mean_store
        epoch_val_loss = self.val_loss_mean_store
        num_epochs = len(epoch_train_loss)
        epochs = [i+1 for i in range(num_epochs)]

        plt.figure(figsize=(10,10))
        plt.plot(epochs, epoch_train_loss, '-', linewidth = 2, color = 'red', label = 'Train Loss')
        plt.plot(epochs, epoch_val_loss, '-', linewidth = 2, color = 'green', label = 'Val Loss')


        plt.grid(visible=True, axis='both', linestyle='-', color='gray', linewidth=0.6)
        min_loss = 0
        max_loss = 1
        plt.ylim(min_loss , max_loss)

        y_ticks = np.arange(min_loss, max_loss, 0.05)
        plt.yticks(y_ticks)
        x_ticks = np.arange(0, num_epochs+5, 5)
        plt.xticks(x_ticks)

        plt.title('Train and Val Loss')
        plt.legend()

        plt.show()
            
    
    def evaluate(self, val_data):
        '''
        Computes the average loss across all validation batches.
        '''
        total_val_loss = 0
        num_val_batches = len(val_data)
        
        # FIX: Loop through the validation batches exactly like you do in training
        with torch.no_grad():
            for bts in range(num_val_batches):
                batch = val_data[bts]
                x_val = batch[0]
                y_val = batch[1]
                total_val_loss += self.loss_fn(batch)
                
        return total_val_loss / num_val_batches
    

    
        