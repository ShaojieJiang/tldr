#!/usr/bin/env python3

# Copyright (c) Facebook, Inc., its affiliates and Shaojie Jiang.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.torch_generator_agent import TorchGeneratorModel


class TransformerModel(TorchGeneratorModel):
    """Transformer model."""

    def __init__(
        self,
        num_words,
        embeddingsize,
        ffn_size,
        nlayers=6,
        nheads=8,
        npos=1024,
        dropout=0,
        lookuptable='unique',
        decoder='same',
        numsoftmax=1,
        attention_length=48,
        padding_idx=0,
        start_idx=1,
        end_idx=2,
        unknown_idx=3,
        input_dropout=0,
        longest_label=1,
        activation='relu',
        sparse=False,
        learn_position=False,
    ):
        super().__init__(
            padding_idx=padding_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            unknown_idx=unknown_idx,
            input_dropout=input_dropout,
            longest_label=longest_label,
        )

        self.encoder = TransformerModule(
            num_words,
            embeddingsize,
            ffn_size,
            padding_idx=padding_idx,
            nlayers=nlayers,
            nheads=nheads,
            npos=npos,
            activation=activation,
            dropout=dropout,
            shared_lt=None,
            sparse=sparse,
            learn_position=learn_position,
        )

        shared_lt = (
            (self.encoder.lt, self.encoder.pe)  # share embeddings between encoder and decoder
            if lookuptable in ('enc_dec', 'all')
            else None
        )
        self.decoder = TransformerModule(
            num_words,
            embeddingsize,
            ffn_size,
            padding_idx=padding_idx,
            nlayers=nlayers,
            nheads=nheads,
            npos=npos,
            activation=activation,
            dropout=dropout,
            shared_lt=shared_lt,
            sparse=sparse,
            learn_position=learn_position,
            is_decoder=True,
        )

        shared_weight = (
            self.decoder.lt  # use embeddings for projection
            if lookuptable in ('dec_out', 'all')
            else None
        )
        self.output = OutputLayer(
            num_words,
            embeddingsize,
            padding_idx=padding_idx,
            dropout=dropout,
            shared_lt=shared_weight,
            sparse=sparse,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 1, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        return None


class TransformerModule(nn.Module):
    "A wrapper for both TransformerEmcoder and TransformerDecoder."

    def __init__(
        self,
        num_words,
        embeddingsize,
        ffn_size,
        padding_idx=0,
        nlayers=2,
        nheads=2,
        npos=1024,
        activation='relu',
        dropout=0.1,
        shared_lt=None,
        sparse=False,
        learn_position=False,
        is_decoder=False,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layers = nlayers
        self.pad_idx = padding_idx
        self.is_decoder = is_decoder

        if shared_lt is None:
            self.lt = nn.Embedding(
                num_words, embeddingsize, padding_idx=padding_idx, sparse=sparse
            )
            self.pe = PositionalEncoding(embeddingsize, dropout=0.0, max_len=npos, learnable=learn_position)
        else:
            self.lt, self.pe = shared_lt

        if is_decoder:
            layer_class = nn.TransformerDecoderLayer
            module_class = nn.TransformerDecoder
        else:
            layer_class = nn.TransformerEncoderLayer
            module_class = nn.TransformerEncoder

        transformer_layer = layer_class(
            embeddingsize, nheads,
            dim_feedforward=ffn_size, dropout=dropout,
            activation=activation)
        self.transformer = module_class(transformer_layer, nlayers) # TODO: norm?


    def forward(self, tensor, enc_states=None, dummy=None):
        emb = self.lt(tensor).transpose(0, 1)
        emb = self.pe(emb)

        # if not self.is_decoder:
        if enc_states is None: # encoder module
            src_key_padding_mask = tensor == self.pad_idx

            out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
            return out, src_key_padding_mask
        else:
            enc_out, src_key_padding_mask = enc_states
            tgt_mask = self._create_selfattn_mask(tensor)
            tgt_key_padding_mask = tensor == self.pad_idx

            out = self.transformer(
                emb, enc_out, tgt_mask=tgt_mask,
                memory_mask=None, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            return out.transpose(0, 1), None

    def _create_selfattn_mask(self, tensor):
        """
        Create a mask to avoid attending to future token
        """
        length = tensor.size(1)
        mask = (torch.tril(tensor.new_ones(length, length)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class OutputLayer(nn.Module):
    """A wrapper of nn.TransformerDecoder"""

    def __init__(
        self,
        num_words,
        embeddingsize,
        padding_idx=0,
        dropout=0.1,
        shared_lt=None,
        sparse=False,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        if shared_lt is None:
            self.lt = nn.Embedding(
                num_words, embeddingsize, padding_idx=padding_idx, sparse=sparse
            )
        else:
            self.lt = shared_lt

    def forward(self, dec_out):
        output = F.linear(dec_out, self.lt.weight)
        output = self.dropout(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, embeddingsize, dropout=0.0, max_len=5000, learnable=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embeddingsize)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embeddingsize, 2).float() * (-math.log(10000.0) / embeddingsize))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        if learnable:
            self.register_parameter('pe', nn.Parameter(pe))
        else:
            self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)