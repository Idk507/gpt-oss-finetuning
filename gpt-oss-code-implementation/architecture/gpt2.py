import torch 
import torch.nn as nn
from torch.nn import functional as F

GPT_CONFIG = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_head': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': True,
}


class LayerNorm(nn.Module):
    """
    Layer normalization module that normalizes the input tensor across the last dimension. It computes the mean and variance of the input tensor, normalizes it, and applies learnable scaling and bias parameters. The normalization is done using the formula:
    norm = (x - mean) / sqrt(var + eps)
    why we are taking the last dimension for normalization is that in many deep learning architectures, especially in transformer models, the last dimension often represents the feature dimension (e.g., embedding size). Normalizing across this dimension helps stabilize the learning process and improves convergence. By normalizing across the feature dimension, we ensure that each feature contributes equally to the model's output, preventing any single feature from dominating the learning process. This is particularly important in models like GPT-2, where the input consists of sequences of tokens represented as embeddings.
    """
    def __init__(self, embed, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed))
        self.bias = nn.Parameter(torch.zeros(embed))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim = -1,keepdim=True) # mean -> dim = -1, keepdim = True where dim is the last dimension of the input tensor x. This computes the mean across the last dimension while keeping the same number of dimensions in the output tensor. Example: If x has shape (batch_size, seq_length,embed_dim), then mean will have shape (batch_size, seq_length, 1) where 1 is the size of the last dimension after computing the mean. 
        var = x.var(dim = -1,unbiased=False,keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return norm * self.weight + self.bias 


