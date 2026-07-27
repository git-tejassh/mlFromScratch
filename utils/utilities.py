import time
import numpy as np
import torch
from torch import nn

def add_to_class(Class):
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    
    return wrapper

class HyperParameters:
    def save_hyperparams(self, ingore = []):
        raise NotImplemented