import torch
from torch import nn

"""
Layer Normalization (LayerNorm) for stabilizing and accelerating training.
Normalizes inputs across the last dimension (features) with learnable scale (γ) and shift (β).

Args:
    d_model (int): Feature dimension of the input (i.e., hidden size).
    eps (float, optional): Small value to avoid division by zero. Default: 1e-12.

Input Shape:
    x: (batch_size, seq_len, d_model) or (batch_size, d_model)

Output Shape:
    Same as input.

Math:
    1. Compute mean and variance over the last dimension:
        μ = mean(x, dim=-1), σ² = var(x, dim=-1)
    2. Normalize:
        x̂ = (x - μ) / √(σ² + eps)
    3. Scale and shift:
        out = γ * x̂ + β

Note:
    - γ (gamma) and β (beta) are learnable parameters initialized to 1 and 0, respectively.
    - Normalization is applied independently per sample and position (if seq_len exists).
"""

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # Step 1: compute mean and variance
        mean = x.mean(-1, keepdim=True)# '-1' means last dimension
        var = x.var(-1, keepdim=True, unbiased=False)

        # Step 2: normlaization
        out = (x - mean) / torch.sqrt(var + self.eps)

        # Setp 3: scale and shift
        out = self.gamma * out + self.beta
        return out