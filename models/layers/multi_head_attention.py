from torch import nn
from models.layers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # Step 1: Linear projections
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # Step 2: Split into multiple heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # Step 3: Scaled dot-product attention
        out, attn_weights = self.attention(q, k, v, mask)

        # Step 4: Concatenate and pass through final linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out, attn_weights

    def split(self, tensor):
        """
        Split tensor into multiple heads.

        :param tensor: [batch_size, seq_len, d_model]
        :return: [batch_size, num_heads, seq_len, d_tensor]
        """
        batch_size, seq_len, _ = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.d_head).transpse(1, 2)
        return tensor
    
    def concat(self, tensor):
        """
        Combine heads back to original dimension.

        :param tensor: [batch_size, num_heads, seq_len, d_tensor]
        :return: [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, _ = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return tensor
