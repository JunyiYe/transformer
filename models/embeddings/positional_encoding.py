import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer models.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model.
            max_len (int, optional): Maximum length of the input sequence. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()

        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # pe's shape (max_len, 1, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for positional encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[: x.size(0), :]
