"""
"Positional or Positionwise" does not mean that it uses positional embeddings 
or positional encodings as inputs (though those are added elsewhere in the Transformer).

Instead, it means that for a sequence of tokens, each token's embedding (at each position) 
is processed through the same feedforward network, independently.
"""

from torch import nn


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN (Feed Forward Network) in the Transformer model.
    
    Args:
        d_model: The number of expected features in the input (input dimension).
        d_ff: The number of features in the feedforward network (hidden dimension).
        dropout: Dropout probability. Default is 0.1.

    Output:
        x: Output tensor of shape (batch_size, seq_len, d_model).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU() # Can also use nn.GELU() or other activation functions
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x