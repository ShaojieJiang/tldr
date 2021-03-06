# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implements NN code for Reformers.

Original paper: https://arxiv.org/abs/1706.03762. (Vaswani, 2017). The
`Annotated Reformer` (Rush, 2018) is an excellent reading guide which explains
much of the mechanics of the Reformer model
(http://nlp.seas.harvard.edu/2018/04/03/attention.html).

This module also supports special segments (ala BERT;
https://arxiv.org/abs/1810.04805), and a few different variations seen in the
literature (BERT and XLM; https://arxiv.org/abs/1901.07291).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    warn_once("Installing APEX can give a significant speed boost.")
    from torch.nn import LayerNorm

LAYER_NORM_EPS = 1e-5  # Epsilon for layer norm.


def _normalize(tensor, norm_layer):
    """Broadcast layer norm."""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


def _create_embeddings(dictionary, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


def _build_encoder(
    opt,
    dictionary,
    embedding=None,
    padding_idx=None,
    reduction_type='mean',
    n_positions=1024,
    n_segments=0,
    rnn_class=nn.GRU,
    full_recurrence=False,
):
    return ReformerEncoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        hiddensize=opt['hiddensize'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        reduction_type=reduction_type,
        n_positions=n_positions,
        n_segments=n_segments,
        activation=opt['activation'],
        variant=opt['variant'],
        output_scaling=opt['output_scaling'],
        rnn_class=rnn_class,
        full_recurrence=full_recurrence,
    )


def _build_decoder(
    opt, dictionary, embedding=None, padding_idx=None, n_positions=1024, n_segments=0, rnn_class=nn.GRU,
    full_recurrence=False,
):
    return ReformerDecoder(
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers'],
        embedding_size=opt['embedding_size'],
        hiddensize=opt['hiddensize'],
        vocabulary_size=len(dictionary),
        embedding=embedding,
        dropout=opt['dropout'],
        attention_dropout=opt['attention_dropout'],
        relu_dropout=opt['relu_dropout'],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        n_positions=n_positions,
        activation=opt['activation'],
        variant=opt['variant'],
        n_segments=n_segments,
        rnn_class=rnn_class,
        full_recurrence=full_recurrence,
    )


def get_n_positions_from_options(opt):
    """Determine n_positions from options dict."""
    if opt.get('n_positions'):
        # if the number of positions is explicitly provided, use that
        n_positions = opt['n_positions']
    else:
        # else, use the worst case from truncate
        n_positions = max(
            opt.get('truncate') or 0,
            opt.get('text_truncate') or 0,
            opt.get('label_truncate') or 0,
        )
        if n_positions == 0:
            n_positions = 1024
    return n_positions


def create_position_codes(n_pos, dim, out):
    """Create positional codes and store them in ``out``."""
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
            for pos in range(n_pos)
        ]
    )

    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)
    out.detach_()
    out.requires_grad = False


class ReformerEncoder(nn.Module):
    """
    Reformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of Reformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int hiddensize: the size of the hidden layer in the hid
    :param embedding: an embedding matrix for the bottom layer of the Reformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the hid. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions:
        Size of the position embeddings matrix.
    :param int n_segments:
        Number of segments/lang/sentence embeddings.
    :param activation:
        Type of nonlinear activation. Can be relu or gelu.
    :param variant:
        Which Reformer architecture to use. Could be AIAYN or XLM.
        Future versions may support things like GPT-2, ...
    :param output_scaling:
        Scale the outputs by a given scalar
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        hiddensize,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction_type='mean',
        n_positions=1024,
        activation='relu',
        variant='aiayn',
        n_segments=0,
        output_scaling=1.0,
        rnn_class=nn.GRU,
        full_recurrence=False,
    ):
        super(ReformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.hiddensize = hiddensize
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout = nn.Dropout(p=dropout)
        self.variant = variant
        self.n_segments = n_segments

        self.n_positions = n_positions
        self.out_dim = embedding_size
        self.full_recurrence = full_recurrence
        assert (
            embedding_size % n_heads == 0
        ), 'Reformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            # raise AssertionError(
            #     "This code should not execute. Left here in case we want to enable it."
            # )
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # embedding normalization
        if self.variant == 'xlm':
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)

        # build the model
        if full_recurrence:
            self.layers = ReformerEncoderLayer(
                n_heads,
                embedding_size,
                hiddensize,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
                variant=variant,
                activation=activation,
                rnn_class=rnn_class,
            )
        else:
            self.layers = nn.ModuleList()
            for _ in range(self.n_layers):
                self.layers.append(
                    ReformerEncoderLayer(
                        n_heads,
                        embedding_size,
                        hiddensize,
                        attention_dropout=attention_dropout,
                        relu_dropout=relu_dropout,
                        dropout=dropout,
                        variant=variant,
                        activation=activation,
                        rnn_class=rnn_class,
                    )
                )
        self.output_scaling = output_scaling

    def forward(self, input, positions=None, segments=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param BoolTensor[batch,seqlen] mask:
            The attention mask; 1 means attend, 0 means ignore.
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        position_embs = self.position_embeddings(positions).expand_as(tensor)
        tensor = tensor + position_embs

        if self.n_segments >= 1:
            if segments is None:
                segments = torch.zeros_like(input)
            tensor = tensor + self.segment_embeddings(segments)

        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)
        # TODO: use encoder hid to initialize!!!
        hid = None
        if self.full_recurrence:
            # reuse the same layer for n_layer times
            for _ in range(self.n_layers):
                tensor, hid = self.layers(tensor, hid, mask)
        else:
            # different layer each time
            for i in range(self.n_layers):
                tensor, hid = self.layers[i](tensor, hid, mask)

        tensor *= self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :]
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0]
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output
        elif self.reduction_type is None or 'none' in self.reduction_type:
            output = tensor
            ret = (output, mask)
            if self.reduction_type == 'none_with_pos_embs':
                ret = (output, mask, position_embs)
            return ret
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )


class ReformerEncoderLayer(nn.Module):
    """Implements a single Reformer encoder layer."""

    def __init__(
        self,
        n_heads,
        embedding_size,
        hiddensize,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant=None,
        rnn_class=nn.GRU,
    ):
        super().__init__()
        self.dim = embedding_size
        self.hid_dim = hiddensize
        self.activation = activation
        self.variant = variant
        self.attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout  # --attention-dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.rnn = ReformerRNN(embedding_size, hiddensize, rnn_class=rnn_class, batch_first=True, dropout=dropout)
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, hid, mask):
        """Forward pass."""
        # tensor = tensor + self.dropout(self.attention(tensor, mask=mask))
        tensor = self.dropout(self.attention(tensor, mask=mask))
        tensor = _normalize(tensor, self.norm1)
        out, hid = self.rnn(tensor, hid)
        # tensor = tensor + self.dropout(out)
        tensor = self.dropout(out)
        tensor = _normalize(tensor, self.norm2)
        tensor *= mask.unsqueeze(-1).type_as(out)
        return tensor, hid


class ReformerDecoder(nn.Module):
    """
    Reformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of Reformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int hiddensize: the size of the hidden layer in the hid
    :param embedding: an embedding matrix for the bottom layer of the Reformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the hid. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        hiddensize,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
        variant='aiayn',
        activation='relu',
        rnn_class=nn.GRU,
        full_recurrence=False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.hiddensize = hiddensize
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.activation = activation
        self.variant = variant
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.n_positions = n_positions
        self.out_dim = embedding_size
        assert (
            embedding_size % n_heads == 0
        ), 'Reformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding
        self.full_recurrence = full_recurrence

        if self.variant == 'xlm':
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        if full_recurrence:
            self.layers = ReformerDecoderLayer(
                n_heads,
                embedding_size,
                hiddensize,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
                activation=activation,
                variant=variant,
                rnn_class=rnn_class,
            )
        else:
            self.layers = nn.ModuleList()
            for _ in range(self.n_layers):
                self.layers.append(
                    ReformerDecoderLayer(
                        n_heads,
                        embedding_size,
                        hiddensize,
                        attention_dropout=attention_dropout,
                        relu_dropout=relu_dropout,
                        dropout=dropout,
                        activation=activation,
                        variant=variant,
                        rnn_class=rnn_class,
                    )
                )

    def forward(self, input, encoder_state, incr_state=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            Ignored. Should always be ``None`` in this version.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout
        hid = None
        if self.full_recurrence:
            # reuse the same layer
            for _ in range(self.n_layers):
                tensor, hid = self.layers(tensor, hid, encoder_output, encoder_mask)
        else:
            # use a different layer each time
            for layer in self.layers:
                tensor, hid = layer(tensor, hid, encoder_output, encoder_mask)

        return tensor, hid


class ReformerDecoderLayer(nn.Module):
    """
    Implements a single Reformer decoder layer.

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a casaul (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        n_heads,
        embedding_size,
        hiddensize,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant='aiayn',
        rnn_class=nn.GRU,
    ):
        super().__init__()
        self.dim = embedding_size
        self.hid_dim = hiddensize
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.rnn = ReformerRNN(embedding_size, hiddensize, rnn_class=rnn_class, batch_first=True, dropout=dropout)
        self.norm3 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def forward(self, x, hid, encoder_output, encoder_mask):
        """Forward pass."""
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        # residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        # x = x + residual
        x = _normalize(x, self.norm1)

        # residual = x
        x = self.encoder_attention(
            query=x, key=encoder_output, value=encoder_output, mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        # x = residual + x
        x = _normalize(x, self.norm2)

        # finally the hid
        # residual = x
        x, hid = self.rnn(x, hid)
        x = self.dropout(x)
        # x = residual + x
        x = _normalize(x, self.norm3)

        return x, hid

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class ReformerGeneratorModel(TorchGeneratorModel):
    """Implements a full generator model, with one encoder and one decoder."""

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        rnn_class = opt.get('rnn_class', 'gru')

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0,
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024
        n_segments = opt.get('n_segments', 0)

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        if opt.get('share_word_embeddings', True):
            encoder_embeddings = self.embeddings
        else:
            encoder_embeddings = None

        self.encoder = _build_encoder(
            opt,
            dictionary,
            encoder_embeddings,
            self.pad_idx,
            reduction_type=None,
            n_positions=n_positions,
            n_segments=n_segments,
            rnn_class=rnn_class,
            full_recurrence=opt['full_recurrence'],
        )
        self.decoder = _build_decoder(
            opt, dictionary, self.embeddings, self.pad_idx, n_positions=n_positions,rnn_class=rnn_class,
            full_recurrence=opt['full_recurrence'],
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder the decoder incremental state.

        Not implemented in Reformers, since ``incremental_state`` is always None.
        """
        # no support for incremental decoding at this time
        return None

    def output(self, tensor):
        """Compute output logits."""
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        return output


class BasicAttention(nn.Module):
    """Implements simple/classical attention."""

    def __init__(self, dim=1, attn='cosine', residual=False, get_weights=True):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        if attn == 'cosine':
            self.cosine = nn.CosineSimilarity(dim=dim)
        self.attn = attn
        self.dim = dim
        self.get_weights = get_weights
        self.residual = residual

    def forward(self, xs, ys, mask_ys=None, values=None):
        """Compute attention.

        Attend over ys with query xs to obtain weights, then apply weights to
        values (ys if yalues is None)

        Args:
            xs: B x query_len x dim (queries)
            ys: B x key_len x dim (keys)
            mask_ys: B x key_len (mask)
            values: B x value_len x dim (values); if None, default to ys
        """
        bsz = xs.size(0)
        y_len = ys.size(1)
        x_len = xs.size(1)
        if self.attn == 'cosine':
            l1 = self.cosine(xs, ys).unsqueeze(self.dim - 1)
        else:
            l1 = torch.bmm(xs, ys.transpose(1, 2))
            if self.attn == 'sqrt':
                d_k = ys.size(-1)
                l1 = l1 / math.sqrt(d_k)
        if mask_ys is not None:
            attn_mask = (mask_ys == 0).view(bsz, 1, y_len)
            attn_mask = attn_mask.repeat(1, x_len, 1)
            l1.masked_fill_(attn_mask, -float('inf'))
        l2 = self.softmax(l1)
        if values is None:
            values = ys
        lhs_emb = torch.bmm(l2, values)

        # # add back the query
        if self.residual:
            lhs_emb = lhs_emb.add(xs)

        if self.get_weights:
            return lhs_emb.squeeze(self.dim - 1), l2
        else:
            return lhs_emb.squeeze(self.dim - 1)


class MultiHeadAttention(nn.Module):
    """
    Implements MultiHeadAttention; this is the core workhorse of the Reformer.

    See Vaswani (2017) for an extensive description.
    """

    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        """Forward pass."""
        # TODO: there are a lot of parameters to document here.

        # Input is [B, query_len, dim]
        # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
        batch_size, query_len, dim = query.size()
        assert (
            dim == self.dim
        ), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                .contiguous()
                .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, key_len)
            .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out


class ReformerRNN(nn.Module):
    "A wraper class for rnn. Conducts reshape before and after."

    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, embedding_size, hiddensize, rnn_class, batch_first=True, dropout=0.0):
        super(ReformerRNN, self).__init__()
        rnn_class = ReformerRNN.RNN_OPTS[rnn_class]
        self.rnn = rnn_class(embedding_size, embedding_size, batch_first=batch_first, dropout=dropout)
        self.emb_size = embedding_size
        self.hid_size = embedding_size
        self.batch_first = batch_first
        # TODO: self.linear = nn.Linear(hiddensize, embedding_size)
        # nn.init.xavier_uniform_(self.rnn.weight)

    def forward(self, x, hid=None):
        # flatten input and hid
        if hid is None:
            hid = x.new_zeros(x.size(0), x.size(1), self.hid_size)
            if type(self.rnn) is nn.LSTM:
                hid = (hid, hid)

        if self.batch_first:
            input = x.view(-1, 1, self.emb_size)
        else:
            input = x.view(1, -1, self.emb_size)
        if type(self.rnn) is nn.LSTM:
            hidden = (hid[0].view(1, -1, self.hid_size), hid[1].view(1, -1, self.hid_size))
        else:
            hidden = hid.view(1, -1, self.hid_size)

        # feed into rnn
        out, hidden = self.rnn(input, hidden)

        # reshape back
        out = out.view(x.size())
        if type(self.rnn) is nn.LSTM:
            hid = (hidden[0].view(hid[0].size()), hidden[1].view(hid[1].size()))
        else:
            hid = hidden.view(hid.size())
        return out, hid


