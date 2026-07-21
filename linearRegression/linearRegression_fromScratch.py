import torch
import numpy as np
import matplotlib.pyplot as plt
import random


class LinearRegressionFromScratch():
  def __init__(self, num_inputs, lr, sigma = 0.01,):
    self.trained = False
    self.lr = lr
    self.sigma = sigma
    self.num_inputs = num_inputs

    if not self.trained:
      self.w = torch.normal(0, self.sigma, (self.num_inputs , 1), requires_grad=True)
      self.b = torch.zeros(1, requires_grad=True)
    else:
      self.w = self.w_final
      self.b = self.b_final

  def calc(self, X):
    return(torch.matmul(X, self.w) + self.b)

  def loss_fn(self, y, y_hat):
    se = (y - y_hat)**2 / 2
    return se.mean()

  def config_optimizer(self):
    return SGD([self.w , self.b] , self.lr)


  def train(self, train_data, val_data, epochs):

    optimizer = self.config_optimizer()
    batch = train_data[0]
    b_size = len(batch)

    self.loss_mean_store = []
    self.val_loss_mean_store = []
    loss_mean = 0
    total_loss = 0
    total_val_loss = 0
    ##ingesting the data, converting it back to x and y - but now in vector format
    num_bts = len(train_data)
    for epoch in range(epochs):
      print('-'*10 , epoch+1, '-'*10)
      for bts in range(num_bts):
        batch = train_data[bts]
        x = batch[0]
        y = batch[1]

        y_hat = self.calc(x)
        loss = self.loss_fn(y, y_hat)
        loss.backward()
        new_params = optimizer.step()

        reset_weights = optimizer.zero_grad()
        loss_mean += loss

      self.loss_mean_store.append(loss_mean)
      loss_mean = loss_mean/num_bts
      total_loss += loss_mean
      val_loss_mean = self.evaluate(val_data)
      self.val_loss_mean_store.append(val_loss_mean)
      total_val_loss += val_loss_mean
      print(f'Mean Train Loss in Epoch: {epoch + 1} is: {loss_mean}')
      print(f'Mean Val Loss in Epoch: {epoch + 1} is: {val_loss_mean}')



    self.trained = True
    average_loss = total_loss / epochs
    average_val_loss = total_val_loss / epochs
    self.w_final = self.w
    self.b_final = self.b
    print('='*40)
    print('\n')
    print(f'final train loss: {loss_mean} \nfinal val loss: {val_loss_mean}')
    print(f'final weights: {self.w.detach().numpy()} \nfinal bias: {self.b.detach().numpy()}')
    # print(f'Average Loss: {average_loss}')
    # print(f'Average Val Loss:' {average_val_loss})






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
    x = torch.tensor(x)
    y = self.calc(x)
    return y.detach().tolist()



  @torch.no_grad()
  def plot(self,):
    epoch_train_loss = self.loss_mean_store
    epoch_val_loss = self.val_loss_mean_store
    num_epochs = len(epoch_train_loss)
    epochs = [i+1 for i in range(num_epochs)]

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

class SGD():
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr

  def step(self):
    with torch.no_grad():
      for param in self.params:
        if param.grad is not None:
          param -= self.lr * param.grad

    return self.params

  def zero_grad(self):
    for param in self.params:
      if param.grad is not None:
        param.grad.zero_()

