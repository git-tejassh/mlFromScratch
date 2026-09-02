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
    """Provide access to the Fashion-MNIST training and validation datasets. The class also supports batching, label conversion, and image visualization.

    Args:
    path: Directory where the dataset files are stored or downloaded.
    """
    def __init__(self, path):
        """Load the Fashion-MNIST training and validation datasets. The datasets
        are prepared for use with machine learning models.

        Args:
            path: Directory where the dataset files are stored or downloaded.

        Returns:
            None.
        """
        self.root = path
        self.trans = transforms.Compose([transforms.Resize(32), 
                                         transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(root = self.root, train = True, transform=self.trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(root=self.root, train = False, transform=self.trans, download=True)
        
    def text_labels(self, indices):
        """Convert numeric class indices into human-readable clothing labels. The
        returned labels correspond to the standard Fashion-MNIST class names.

        Args:
            indices: Numeric class indices to convert.

        Returns:
            A list of human-readable labels.
        """
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        
        return [labels[int(i)] for i in indices]
    
    def get_data(self, train, batch_size = 16):
        """Create a data loader for the training or validation dataset. The loader
        returns data in batches and shuffles it when training data is selected.

        Args:
            train: Whether to load the training dataset instead of the validation
                dataset.
            batch_size: Number of samples in each batch.

        Returns:
            A data loader for the selected dataset.
        """
        self.batch_size = batch_size
        data = self.train if train else self.val
        return DataLoader(data, self.batch_size , shuffle=train)
    

    def show_images(self, imgs, num_rows, num_cols, titles=None, scale=1.5):
        """Display a collection of images in a grid. Optional titles can be shown
        above individual images.

        Args:
            imgs: Images to display.
            num_rows: Number of rows in the image grid.
            num_cols: Number of columns in the image grid.
            titles: Optional titles for the displayed images.
            scale: Scale factor used to determine the figure size.

        Returns:
            The axes used to display the images."""
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
        """Display a batch of images with optional or inferred labels. The images
        are arranged in a grid with the specified number of rows and columns.

        Args:
            batch: A pair containing images and their corresponding labels.
            nrows: Number of rows in the image grid.
            ncols: Number of columns in the image grid.
            labels: Optional labels to display with the images.

        Returns:
            None.
        """
        """Visualize a batch of data."""
        x, y = batch
        if not labels:
            labels = self.text_labels(y)
        # Notice 'titles=labels' instead of '[labels]' to avoid nesting lists
        self.show_images(x.squeeze(1), nrows, ncols, titles=labels)


    
    
        

