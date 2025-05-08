from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.positionwise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Step 1: Compute self attention
        _x = x
        x = self.attn(q=x, k=x, v=x, mask=mask)

        # Step 2: Addition and normalization
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # Step 3: Positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # Step 4: Addition and normalization
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x