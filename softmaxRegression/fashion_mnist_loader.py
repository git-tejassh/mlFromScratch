import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import sys
sys.path.append("C:/Users/tjsss/OneDrive/Desktop/machine learning/mlFromScratch")
import torchvision
from torchvision import transforms
from utils.utilities import add_to_class, HyperParameters
from utils.crossEntropyLoss import CrossEntropyLoss

##loading MNIST DATASET

class FashionMNIST():
    def __init__(self, path):
        self.root = path
        self.trans = transforms.Compose([transforms.Resize(32), 
                                         transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(root = self.root, train = True, transform=self.trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(root=self.root, train = False, transform=self.trans, download=True)
        
    def text_labels(self, indices):
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        
        return [labels[int(i)] for i in indices]
    
    def get_data(self, train, batch_size = 16):
        self.batch_size = batch_size
        data = self.train if train else self.val
        return DataLoader(data, self.batch_size , shuffle=train)
    

    def show_images(self, imgs, num_rows, num_cols, titles=None, scale=1.5):
        """Plot a list of images."""
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            try:
                img = img.numpy()
            except:
                pass
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles and i < len(titles):
                ax.set_title(titles[i])
        plt.tight_layout()
        return axes


    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        """Visualize a batch of data."""
        x, y = batch
        if not labels:
            labels = self.text_labels(y)
        # Notice 'titles=labels' instead of '[labels]' to avoid nesting lists
        self.show_images(x.squeeze(1), nrows, ncols, titles=labels)


    
    
        

