import torch.nn as nn

from models.embeddings.transformer_embedding import TransformerEmbedding
from models.blocks.decoder_layer import DecoderLayer

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder Module.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the model (input and output).
        num_heads (int): Number of attention heads.
        d_ffn (int): Dimension of the feedforward network.
        num_layers (int): Number of decoder layers.
        max_len (int, optional): Maximum length of the input sequence. Default is 5000.
        padding_idx (int, optional): Padding index for the input sequences. Default is 0.
        dropout (float, optional): Dropout probability. Default is 0.1.
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ffn: int, 
                 num_layers: int, max_len: int = 5000, padding_idx: int = 0, 
                 dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, padding_idx, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ffn, dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the decoder.

        Args:
            tgt (torch.Tensor): Target tensor of shape (batch_size, tgt_seq_len).
            enc_output (torch.Tensor): Encoder output tensor of shape (batch_size, src_seq_len, d_model).
            tgt_mask (torch.Tensor, optional): Target mask tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
            memory_mask (torch.Tensor, optional): Memory mask tensor of shape (batch_size, 1, tgt_seq_len, src_seq_len).
        """
        x = self.embedding(tgt)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)

        return self.norm(x) # Layer normalization is recommended in practice