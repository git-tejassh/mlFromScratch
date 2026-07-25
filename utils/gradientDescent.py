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