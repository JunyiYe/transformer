import torch.nn as nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.positionwise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    
    Args:
        d_model (int): Dimension of the model (input and output).
        d_ff (int): Dimension of the feedforward network.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Default is 0.1.

    Output:
        x (Tensor): Output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Step 1: Compute self-attention
        attn_output, _ = self.attn(Q=x, K=x, V=x, mask=mask)

        # Step 2: Residual connection and LayerNorm
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Step 3: Positionwise feed forward network
        ffn_output = self.ffn(x)

        # Step 4: Addition and normalization
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x