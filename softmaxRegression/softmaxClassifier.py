from utils.gradientDescent import SGD
import torch
import torch.nn as nn
from utils.crossEntropyLoss import CrossEntropyLoss
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    
    def __init__(self, input_dims ,n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.linear = nn.Linear(input_dims*input_dims ,self.n_classes)
        self.bias = self.linear.bias
        self.weight = self.linear.weight

        self.criterion = CrossEntropyLoss()

    def configure_optim(self, ):
        return SGD([self.weight , self.bias], lr = 0.01)
    @torch.enable_grad()        
    def calc(self, x):
        x = x.view(x.size(0) , -1)
        return self.linear(x)
        
    @torch.enable_grad()
    def loss_fn(self,x, y):
        logits = self.calc(x)
        return self.criterion.forward(logits, y) 
    
    def accuracy(self, logits , y):
        predictions = torch.argmax(logits, dim = 1)
        return (predictions == y).float().mean()
    
    @torch.enable_grad()
    def train(self, train_loader, val_loader, epochs):
        optim = self.configure_optim()
        self.loss_mean_store = []
        self.val_loss_mean_store  =[]
        
        
        for epoch in range(epochs):
            
            mean_train_loss = 0.0
            running_loss = 0.0
            num_train_batches = 0
            print('-'*10 , epoch+1, '-'*10)
            
            for x_batch, y_batch in train_loader:
                optim.zero_grad()
                loss = self.loss_fn(x_batch, y_batch.squeeze(-1).long())
                loss.backward()
                optim.step()
                running_loss += loss.item()
                num_train_batches += 1
            
            mean_train_loss = running_loss / num_train_batches
            self.loss_mean_store.append(mean_train_loss)
            
            ## VALIDATION LOSS
            running_val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad(): # Ensure gradient history tracking is shut down
                for x_val_batch, y_val_batch in val_loader:
                    # Execute validation calculations using the step helper
                    val_loss, val_acc = self.validation_step(x_val_batch, y_val_batch)
                    running_val_loss += val_loss
                    num_val_batches += 1
                    
            mean_val_loss = running_val_loss / num_val_batches
            self.val_loss_mean_store.append(mean_val_loss)
            
            print(f'Mean Train Loss in Epoch: {epoch + 1} is: {mean_train_loss:.4f}')
            print(f'Mean Val Loss in Epoch: {epoch + 1} is: {mean_val_loss:.4f}')
            
    
    @torch.no_grad()
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
            

    @torch.no_grad()
    def validation_step(self, x , y):
        logits = self.calc(x)
        val_loss = self.loss_fn(x, y.squeeze(-1).long())
        val_acc = self.accuracy(logits, y)
        return (val_loss.mean().item(), val_acc.mean().item())
    
    @torch.no_grad()
    def predict(self, x):
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        logits = self.calc(x.unsqueeze(0))
        prediction = torch.argmax(logits, 1).item()
        plt.figure(figsize=(8, 8)) 
        plt.imshow(x.squeeze(), )
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Prediction: {labels[prediction]}')
        plt.tight_layout()
        plt.show()
        
        
        
    
    