import torch
from torch import nn
import pandas as pd

class BaseModel(nn.Module):   
    def forward(self, x, task):
        x, task = self.normalize(x, task)
        return self._forward(x, task)
    
    def _forward(self, x, task):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def normalize(self, x, task):
        task = (task - self.task_min) / (self.task_max - self.task_min)
        return x / self.nodes_norm, task

    def register_norm(self, norm, dataset: pd.DataFrame= None, device=None):

        self.register_buffer('nodes_norm', torch.tensor(norm, device=device).max(dim=0, keepdim=True).values)
        
        if dataset is not None:
            
            self.register_buffer('task_min', torch.tensor(dataset[["TaskSize", "CyclesPerBit", "TransBitRate", "DDL"]].min().values, dtype=torch.float32, device=device))
            self.register_buffer('task_max', torch.tensor(dataset[["TaskSize", "CyclesPerBit", "TransBitRate", "DDL"]].max().values, dtype=torch.float32, device=device))
            
            print(f"Registered normalization factors: nodes_norm={self.nodes_norm}, task_min={self.task_min}, task_max={self.task_max}")    
        else:
            self.register_buffer('task_min', torch.tensor([0.0], dtype=torch.float32, device=device))
            self.register_buffer('task_max', torch.tensor([1.0], dtype=torch.float32, device=device))
        