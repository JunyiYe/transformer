import torch.nn as nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.positionwise_feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Decoder Layer in the Transformer model.

    Args:
        d_model (int): The number of expected features in the input (input dimension).
        num_heads (int): Number of attention heads.
        d_ffn (int): The number of features in the feedforward network (hidden dimension).
        dropout (float, optional): Dropout probability. Default is 0.1.
    """
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the decoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, tgt_seq_len, d_model).
            enc_output (Tensor): Encoder output tensor of shape (batch_size, src_seq_len, d_model).
            tgt_mask (Tensor, optional): Target mask for self-attention.
            memory_mask (Tensor, optional): Memory mask for cross-attention.
        """
        # 1. Masked self-attention
        self_attn_output, _ = self.self_attn(Q=x, K=x, V=x, mask=tgt_mask)
        x = x + self.dropout1(self_attn_output)
        x = self.norm1(x)

        # 2. Cross-attention with encoder output
        cross_attn_output = self.cross_attn(Q=x, K=enc_output, V=enc_output, mask=memory_mask)
        x = x + self.dropout2(cross_attn_output)
        x = self.norm2(x)

        # 3. Feedforward network
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)

        return x
        