import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import sys


sys.path.append("C:/Users/tjsss/OneDrive/Desktop/machine learning/mlFromScratch")


from utils.gradientDescent import SGD



class LogisticRegressionFromScratch():
    
    def __init__ (self, num_feats ,lr = 0.001 ,sigma = 0.01):
        self.trained = False
        self.lr = lr
        self.num_feats = num_feats
        self.sigma = sigma
    
        if not self.trained:
            self.w = torch.normal(0, self.sigma, (self.num_feats , 1), requires_grad=True)
            self.b = torch.zeros(1, requires_grad=True)
        else:
            self.w = self.w_final
            self.b = self.b_final
        
    def calc(self, x):
        z = (torch.matmul(x, self.w) + self.b)
        y = ( 1 / (1 + torch.exp(-z)))
        return y
    
    def loss_fn(self,y_hat, y):
        l = -(y*torch.log(y_hat) + (1-y)*torch.log(1-y_hat))
        return l.mean()
    
    def config_optimizer(self,):
        return SGD([self.w, self.b], lr=self.lr)
        
    
    def train(self, train_data, val_data, epochs):
        optimizer = self.config_optimizer()
        batch = train_data[0]
        b_size = len(batch)
        
        self.loss_mean_list = []
        val_loss_mean = 0.0
        self.val_loss_mean_list = []
        
        num_bts = len(train_data)
        for epoch in range(epochs):
            loss_mean = 0.0
            print('-'*10 , epoch+1, '-'*10)
            for bts in range(num_bts):
                batch = train_data[bts]
                x = batch[0]
                y = batch[1]
                
                y_hat = self.calc(x)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_mean += loss
            loss_mean_mean = loss_mean / num_bts
            self.loss_mean_list.append(loss_mean_mean.item())
            
            val_loss_mean = self.evaluate(val_data)
            self.val_loss_mean_list.append(val_loss_mean)
            
            print(f'Mean Train Loss in Epoch: {epoch + 1} is: {loss_mean_mean}')
            print(f'Mean Val Loss in Epoch: {epoch + 1} is: {val_loss_mean}')      
            
        self.trained = True
        self.w_final = self.w
        self.b_final = self.b
        print('='*40)
        print('\n')
        print(f'final train loss: {loss_mean} \nfinal val loss: {val_loss_mean}')
        print(f'final weights: {self.w.detach().numpy()} \nfinal bias: {self.b.detach().numpy()}') 
        
    @torch.no_grad()
    def evaluate(self, data):
        batch = data[0]
        b_size = len(batch)
    
    
        val_loss_mean = 0
        total_loss = 0
        num_bts = len(data)
        for bts in range(num_bts):
            batch = data[bts]
            x = batch[0]
            y = batch[1]
    
            y_hat = self.calc(x)
            loss = self.loss_fn(y, y_hat)
    
            val_loss_mean += loss
    
        val_loss_mean = val_loss_mean / len(data)
        return val_loss_mean     
                
        
    @torch.no_grad()
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        y = self.calc(x)
        return y.detach().tolist()



    @torch.no_grad()
    def history(self,):
        epoch_train_loss = self.loss_mean_list
        epoch_val_loss = self.val_loss_mean_list
        num_epochs = len(epoch_train_loss)
        epochs = [i+1 for i in range(num_epochs)]

        print(epoch_train_loss, epoch_val_loss,)
        
        plt.figure(figsize=(10,10))
        plt.plot(epochs, epoch_train_loss, '-', linewidth = 2, color = 'red', label = 'Train Loss')
        plt.plot(epochs, epoch_val_loss, '-', linewidth = 2, color = 'green', label = 'Val Loss')


        plt.grid(visible=True, axis='both', linestyle='-', color='gray', linewidth=0.6)
        min_loss = 0
        max_loss = 10
        plt.ylim(min_loss , max_loss)

        y_ticks = np.arange(min_loss, max_loss, 0.25)
        plt.yticks(y_ticks)
        x_ticks = np.arange(0, num_epochs+5, 5)
        plt.xticks(x_ticks)

        plt.title('Train and Val Loss')
        plt.legend()

        plt.show()
        
        
