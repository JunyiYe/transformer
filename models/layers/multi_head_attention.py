from torch import nn
from models.layers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as used in Transformer models.

    Args:
        d_model (int): Dimension of the model (input and output).
        num_heads (int): Number of attention heads.
        Q (Tensor): Query tensor of shape (batch_size, seq_len_q, d_model).
        K (Tensor): Key tensor of shape (batch_size, seq_len_k, d_model).
        V (Tensor): Value tensor of shape (batch_size, seq_len_v, d_model).
        mask (Tensor, optional): Mask tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)

    Output:
        out (Tensor): Output tensor of shape (batch_size, seq_len, d_model).
        attn_weights (Tensor): Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k).
    """
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layr for projecting Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention()

        # Final output projection layer
        self.W_concat = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        # Step 1: Linear projections
        Q, K, V = self.w_q(Q), self.W_k(K), self.W_v(V)

        # Step 2: Split into multiple heads
        Q, K, V = self.split(Q), self.split(K), self.split(V)

        # Step 3: Scaled dot-product attention
        output, attn_weights = self.attention(Q, K, V, mask)

        # Step 4: Concatenate and pass through final linear layer
        output = self.concat(output)
        output = self.w_concat(output)

        return output, attn_weights

    def split(self, tensor):
        """
        Split tensor into multiple heads.

        Args:
            tensor: [batch_size, seq_len, d_model]
        
        Output:
            tensor: [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = tensor.size()
        # PyTorch requires the tensor to be contiguous in memory before reshaping (view operation)
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpse(1, 2)
        
        return tensor
    
    def concat(self, tensor):
        """
        Combine heads back to original dimension.

        Args:
            tensor: [batch_size, num_heads, seq_len, d_tensor]
        
        Output:
            tensor: [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return tensor
