import math
import torch
from torch import nn

"""
Compute scaled dot-product attention

Let:
    - Query (Q): Represents the current focus (e.g., target word in decoder or any word in encoder self-attention).
    - Key (K): Encodes the content of all words to compute relevance with Query.
    - Value (V): Contains the actual information to aggregate based on attention weights.

Input shapes:
    Q: (batch_size, num_heads, seq_len_q, head_dim)
    K: (batch_size, num_heads, seq_len_k, head_dim)
    V: (batch_size, num_heads, seq_len_v, head_dim)

Output shape:
    output: (batch_size, num_heads, seq_len_q, head_dim)
    attn_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
"""

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, num_heads, seq_len_k, head_dim = k.size()

        # Step 1: dot-product Query with Key^T
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(head_dim) # Scaled dot-product

        # Step 2: apply mask (optionaly)
        if mask is not None:
            # elements of 0s will be masked, while 1s will not be masked
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # Step 3: softmax
        attn_weights = self.softmax(scores)

        # Step 4: multiply by Value
        output = torch.matmul(attn_weights, v)

        return output, attn_weights