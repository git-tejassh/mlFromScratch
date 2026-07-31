import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,logits, y):
        y_exp = torch.exp(logits)
        prob = y_exp / torch.sum(y_exp, dim =1, keepdim = True)
        
        batch_size = y_exp.size(0)
        row_indices = torch.arange(batch_size)
        
        true_class_prob = prob[row_indices, y]
        loss = -torch.log(true_class_prob + 1e-15)
        return loss.mean()