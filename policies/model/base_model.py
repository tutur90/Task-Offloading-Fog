import torch
from torch import nn

class BaseModel(nn.Module):   
    def forward(self, x, task):
        x = self.normalize(x, task)
        return self._forward(x, task)
    
    def _forward(self, x, task):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def normalize(self, x, task):
        task = (task - self.task_min) / (self.task_max - self.task_min)
        return x / self.nodes_norm, task

    def register_norm(self, norm, dataset=None):

        self.register_buffer('nodes_norm', torch.tensor(norm).max(dim=0, keepdim=True).values)
        if dataset is not None:
            task_vals = dataset["task_length"]
            self.register_buffer('task_min', torch.tensor([task_vals.min()], dtype=torch.float32))
            self.register_buffer('task_max', torch.tensor([task_vals.max()], dtype=torch.float32))
        else:
            self.register_buffer('task_min', torch.tensor([0.0], dtype=torch.float32))
            self.register_buffer('task_max', torch.tensor([1.0], dtype=torch.float32))
        