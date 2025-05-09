import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds positional encodings to the input embeddings using sine and cosine functions.

    Args:
        d_model (int): The dimension of the model.
        max_len (int, optional): Maximum length of the input sequence. Defaults to 5000.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Returns:
        torch.Tensor: Positional encoding matrix of shape (seq_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # pe's shape (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        # pe automatically broadcasts to (batch_size, x's seq_len, d_model)
        x = x + self.pe[:, x.size(1), :] 

        return self.dropout(x)
