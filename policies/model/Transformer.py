from policies.model.transformer_encoder.encoder import TransformerEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import random


class Transformer(nn.Module):
    def __init__(self, d_in, d_pos, d_model=8, d_ff=8, n_heads=1, n_layers=1, dropout=0.2):
        super().__init__()

        
        self.embed = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(d_pos, d_model))
        self.trasformer_encoder = TransformerEncoder(d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos_embed
        x = self.trasformer_encoder(x, None)
        x = self.fc(x)
        x = self.softmax(x)
        return x
        
    



