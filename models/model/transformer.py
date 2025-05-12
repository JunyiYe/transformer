import torch.nn as nn

from models.model.transformer_encoder import TransformerEncoder
from models.model.transformer_decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    Transformer model that combines both encoder and decoder.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        d_model (int): Dimension of the model (input and output).
        num_heads (int): Number of attention heads.
        d_ffn (int): Dimension of the feedforward network.
        num_layers (int): Number of encoder and decoder layers.
        max_len (int, optional): Maximum length of the input sequence. Default is 5000.
        padding_idx (int, optional): Padding index for the input sequences. Default is 0.
        dropout (float, optional): Dropout probability. Default is 0.1.
    """

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx， src_vocab_size: int, tgt_vocab_size: int, 
                 d_model: int, num_heads: int, d_ffn: int, num_layers: int, max_len: int = 5000, 
                 padding_idx: int = 0, dropout: float = 0.1):
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx

        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, d_ffn,
                                          num_layers, max_len, padding_idx, dropout)

        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, d_ffn,
                                          num_layers, max_len, padding_idx, dropout)    

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    # def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
    #     """
    #     Forward pass for the transformer model.

    #     Args:
    #         src (torch.Tensor): Source input tensor of shape (batch_size, src_seq_len).
    #         tgt (torch.Tensor): Target input tensor of shape (batch_size, tgt_seq_len).
    #         src_mask (torch.Tensor, optional): Source mask tensor of shape (batch_size, 1, 1, src_seq_len).
    #         tgt_mask (torch.Tensor, optional): Target mask tensor of shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
    #         memory_mask (torch.Tensor, optional): Memory mask tensor of shape (batch_size, 1, tgt_seq_len, src_seq_len).
    #     """
    #     # Encode source sequence
    #     memory = self.encoder(src, src_mask)

    #     # Decode with encoder memory
    #     decoder_output = self.decoder(tgt, memory, tgt_mask, memory_mask)

    #     # Project to target vocabulary logits
    #     logits = self.output_projection(decoder_output)

    #     return logits

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask