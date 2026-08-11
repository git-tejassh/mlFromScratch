import torch
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
      
      
class miniBatchGD():
  '''
  function to do mini batch gradient descent in batches
  '''
  
  def __init__(self, params : torch.tensor , batch_size : int , lr : float ):
    '''
    Function to do mini batch GD
    Args:
      params : [torch.tensor] input the parameters of the model
      batch_size : [int] size of the batch
      lr : [float] learning rate
    '''
    
    self.params = params
    self.bs = batch_size
    self.lr = lr
    
  def step(self):
    with torch.no_grad:
      mean_param_grad = 0.0
      for param in self.params:
        if param.grad is not None:
          mean_param_grad += param.grad
        mean_param_grad /= self.bs
        param -= self.lr * mean_param_grad
      return self.params
    
  def zero_grad(self):
    for param in self.params:
      if param.grad is not None:
        param.grad.zero_()