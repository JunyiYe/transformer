from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # Step 1: Compute self_attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

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