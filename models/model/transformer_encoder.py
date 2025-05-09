import torch.nn as nn

from models.embeddings.transformer_embedding import TransformerEmbedding
from models.blocks.encoder_layer import EncoderLayer


class TransformerEncoder(nn.Module):
    """
    Stacked Transformer Encoder consisting of N identical layers.

    Args:
        d_model (int): Dimension of the model (input and output).
        num_heads (int): Number of attention heads.
        d_ffn (int): Dimension of the feedforward network.
        num_layers (int): Number of encoder layers.
        dropout (float, optional): Dropout probability. Default is 0.1.
    """
    def __init__(self, d_model: int, num_heads: int, d_ffn: int, num_layers: int, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.embedding = TransformerEmbedding(d_model, d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        """
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            src_mask (torch.Tensor, optional): Source mask tensor of shape (batch_size, 1, 1, seq_len).
        """
        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x