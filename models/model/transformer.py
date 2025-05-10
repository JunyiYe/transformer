import torch.nn as nn

from models.model.transformer_encoder import TransformerEncoder
from models.model.transformer_decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    Transformer model that combines both encoder and decoder.

    Args:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, num_heads: int, d_ffn: int,
                 num_layers: int, max_len: int = 5000, padding_idx: int = 0, dropout: float = 0.1):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, d_ffn,
                                          num_layers, max_len, padding_idx, dropout)

        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, d_ffn,
                                          num_layers, max_len, padding_idx, dropout)    

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the transformer model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target input tensor of shape (batch_size, tgt_seq_len).
            src_mask (torch.Tensor, optional): Source mask tensor of shape (batch_size, 1, 1, src_seq_len).
            tgt_mask (torch.Tensor, optional): Target mask tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
            memory_mask (torch.Tensor, optional): Memory mask tensor of shape (batch_size, 1, tgt_seq_len, src_seq_len).
        """
        # Encode source sequence
        memory = self.encoder(src, src_mask)

        # Decode with encoder memory
        decoder_output = self.decoder(tgt, memory, tgt_mask, memory_mask)

        # Project to target vocabulary logits
        logits = self.output_projection(decoder_output)

        return logits