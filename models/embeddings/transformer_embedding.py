import torch.nn as nn

from models.embeddings.token_embedding import TokenEmbedding
from models.embeddings.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """
    Initialize the TransformerEmbedding class with a specified model name.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model (input and output).
        max_len (int, optional): Maximum length of the input sequence. Default is 5000.
        padding_idx (int, optional): Padding index. Default is 0.
        dropout (float, optional): Dropout probability. Default is 0.1.
    """
    def __init__(self, vocab_size, d_model, max_len=5000, padding_idx=0, dropout=0.1):
        super(TransformerEmbedding, self).__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.droupout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the Transformer embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.token_embedding(x)
        x = self.positional_encoding(x)

        return self.dropout(x)