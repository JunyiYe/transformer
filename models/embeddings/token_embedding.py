import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer for Transformer models.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model (input and output).
        padding_idx (int, optional): Padding index. Default is 0.

    Returns:
        torch.Tensor: Token embedding matrix of shape (vocab_size, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0) -> None:
        super(TokenEmbedding, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Scale the embeddings by the square root of d_model
        return self.embedding(x) * math.sqrt(self.d_model)