from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.positionwise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads, dropout: float = 0.1):
        """
        Decoder Layer in the Transformer model.

        Args:
            d_model (int): The number of expected features in the input (input dimension).
            d_ffn (int): The number of features in the feedforward network (hidden dimension).
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout probability. Default is 0.1.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        