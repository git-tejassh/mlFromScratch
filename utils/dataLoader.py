import torch
import numpy as np
import matplotlib.pyplot as plt
import random

class data_loader():

  def __init__(self, x, y, split, batch_size):
    self.batch_size = batch_size
    random.shuffle([x,y])

    self.split_data = int(split*len(y))
    self.train_x = x[:self.split_data]
    self.train_y = y[:self.split_data]
    self.test_x = x[self.split_data:]
    self.test_y = y[self.split_data:]



  def train_loader(self,):
    loader = []
    for start in range(0, len(self.train_y), self.batch_size):
      end = start + self.batch_size
      batch_x_data = self.train_x[start:end]
      batch_y_data = self.train_y[start:end]

      # 2. Convert each into its own independent tensor
      batch_x = torch.tensor(batch_x_data, dtype=torch.float32)
      batch_y = torch.tensor(batch_y_data, dtype=torch.float32)

      # 3. Store them as a tuple pair
      loader.append((batch_x, batch_y))

    return loader


  def test_loader(self,):
    loader = []

    for start in range(0, len(self.test_y), self.batch_size):
      end = start + self.batch_size
      batch_x_data = self.test_x[start:end]
      batch_y_data = self.test_y[start:end]

      # 2. Convert each into its own independent tensor
      batch_x = torch.tensor(batch_x_data, dtype=torch.float32)
      batch_y = torch.tensor(batch_y_data, dtype=torch.float32)

      # 3. Store them as a tuple pair
      loader.append((batch_x, batch_y))
    return loader
