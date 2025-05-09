import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Compute scaled dot-product attention

    Let:
        - Query (Q): Represents the current focus (e.g., target word in decoder or any word 
          in encoder self-attention).
        - Key (K): Encodes the content of all words to compute relevance with Query.
        - Value (V): Contains the actual information to aggregate based on attention weights.

    Input shapes:
        Q: (batch_size, num_heads, seq_len_q, head_dim)
        K: (batch_size, num_heads, seq_len_k, head_dim)
        V: (batch_size, num_heads, seq_len_v, head_dim)
        mask: (batch_size, num_heads, seq_len_q, seq_len_k) or (1, 1, eq_len_q, seq_len_k) broadcasted
        Notes: d_model = num_heads * head_dim
               d_q = d_k = d_v = head_dim
               seq_len_k = seq_len_v

    Output shape:
        output: (batch_size, num_heads, seq_len_q, head_dim)
        attn_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = K.size(-1)

        # Step 1: Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)

        # Step 2: Apply mask (optionaly)
        if mask is not None:
            # Elements of 0s will be masked, while 1s will not be masked
            # -inf's softmax becomes zero
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 3: Softmax over the last dimension (seq_len_k)
        attn_weights = F.softmax(scores, dim=-1)

        # Step 4: Multiply with V
        output = torch.matmul(attn_weights, V)

        return output, attn_weights