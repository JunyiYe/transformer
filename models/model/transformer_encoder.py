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
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ffn: int, 
                 num_layers: int, max_len: int = 5000, padding_idx: int = 0, 
                 dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, padding_idx, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        """
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            src_mask (torch.Tensor, optional): Source mask tensor of shape (batch_size, 1, 1, seq_len).
        """
        x = self.embedding(src)

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x) # Layer normalization is recommended in practice