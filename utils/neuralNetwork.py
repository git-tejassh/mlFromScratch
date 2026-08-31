import torch
import numpy as np
import torch.nn as nn
import math


class CustomLinear(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, bias: bool = True) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.bias = bias
        
        self.weight = nn.Parameter(torch.empty(out_feats, in_feats))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_feats))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        #uniform initialization
        bound = 1 / math.sqrt(self.in_feats)
        nn.init.uniform_(self.weight, -bound, bound)
        
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # perform x@W.T + b
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        
        return output
        