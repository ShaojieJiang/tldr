#!/usr/bin/env python3

# Copyright (c) Facebook, Inc., its affiliates and Shaojie Jiang.

import math, copy, os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.core.dict import DictionaryAgent

from transformers import BertTokenizer
from transformers import BertConfig as TransformerConfig
from transformers import BertModel as Transformer


class TransformerModel(TorchGeneratorModel):
    """Transformer model using HuggingFace transformers.BertModel."""

    def __init__(
        self,
        num_words,
        embeddingsize,
        ffn_size,
        use_pretrained=False,
        nlayers=6,
        nheads=8,
        npos=1024,
        dropout=0,
        lookuptable='unique',
        decoder='same',
        numsoftmax=1,
        attn_drop=0,
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
            use_pretrained=use_pretrained,
            padding_idx=padding_idx,
            nlayers=nlayers,
            nheads=nheads,
            npos=npos,
            activation=activation,
            dropout=dropout,
            attn_drop=attn_drop,
            shared_lt=None,
            sparse=sparse,
            learn_position=learn_position
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
            use_pretrained=use_pretrained,
            padding_idx=padding_idx,
            nlayers=nlayers,
            nheads=nheads,
            npos=npos,
            activation=activation,
            dropout=dropout,
            attn_drop=attn_drop,
            shared_lt=shared_lt,
            sparse=sparse,
            learn_position=learn_position,
            is_decoder=True,
        )

        shared_weight = (
            self.decoder.lt  # use embeddings for projection
            if lookuptable in ('dec_out', 'dec_out_copy', 'all')
            else None
        )
        self.output = OutputLayer(
            num_words,
            self.encoder.transformer.embeddings.word_embeddings.embedding_dim,
            padding_idx=padding_idx,
            dropout=dropout,
            shared_lt=shared_weight,
            copy_weight=True if 'copy' in lookuptable else False,
            sparse=sparse,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
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
        use_pretrained=False,
        padding_idx=0,
        nlayers=2,
        nheads=2,
        npos=1024,
        activation='relu',
        dropout=0.1,
        attn_drop=0.1,
        shared_lt=None,
        sparse=False,
        learn_position=False,
        is_decoder=False,
    ):
        super().__init__()

        if not use_pretrained:
            config = TransformerConfig(
                vocab_size_or_config_json_file=num_words,
                hidden_size=embeddingsize,
                num_hidden_layers=nlayers,
                num_attention_heads=nheads,
                intermediate_size=ffn_size,
                hidden_act=activation,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=attn_drop,
                max_position_embeddings=npos,
                is_decoder=is_decoder,
            )

            self.dropout = nn.Dropout(p=dropout)
            self.layers = nlayers
            self.pad_idx = padding_idx
            self.is_decoder = is_decoder

            raise NotImplementedError
            if shared_lt is None:
                self.lt = nn.Embedding(
                    num_words, embeddingsize, padding_idx=padding_idx, sparse=sparse
                )
                self.pe = PositionalEncoding(embeddingsize, dropout=0.0, max_len=npos, learnable=learn_position)
            else:
                self.lt, self.pe = shared_lt

            self.transformer = Transformer(config)
        else: # use pretrained parameters
            self.pad_idx = padding_idx
            config = TransformerConfig.from_pretrained('bert-base-uncased')
            config.is_decoder = is_decoder
            self.transformer = Transformer.from_pretrained('bert-base-uncased', config=config)
            if shared_lt:
                self.transformer.embeddings.word_embeddings = shared_lt[0]
                self.transformer.embeddings.position_embeddings = shared_lt[1]
            self.lt = self.transformer.embeddings.word_embeddings
            self.pe = self.transformer.embeddings.position_embeddings

    def forward(self, tensor, enc_states=None, dummy=None):

        # if not self.is_decoder:
        if enc_states is None: # encoder module
            attn_mask = (tensor != self.pad_idx).float()

            out, *_ = self.transformer(
                input_ids=tensor,
                attention_mask=attn_mask,
            )
            return out, attn_mask
        else:
            enc_out, src_attn_mask = enc_states
            attn_mask = (tensor != self.pad_idx).float()
            out, *_ = self.transformer(
                input_ids=tensor,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_out,
                encoder_attention_mask=src_attn_mask,
            )
            return out, None

class OutputLayer(nn.Module):
    """A wrapper of nn.TransformerDecoder"""

    def __init__(
        self,
        num_words,
        embeddingsize,
        padding_idx=0,
        dropout=0.1,
        shared_lt=None,
        copy_weight=True,
        sparse=False,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        if shared_lt is None:
            self.lt = nn.Embedding(
                num_words, embeddingsize, padding_idx=padding_idx, sparse=sparse
            )
            nn.init.normal_(self.lt.weight, mean=0.0, std=0.02)
        else:
            if copy_weight:
                self.lt = copy.deepcopy(shared_lt)
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

class HfDictionaryAgent(DictionaryAgent):
    "A wrapper for HuggingFace BertTokenizer."

    @staticmethod
    def add_cmdline_args(argparser):
        return DictionaryAgent.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        self.opt = copy.deepcopy(opt)

        self.null_token = opt.get('dict_nulltoken')
        self.start_token = opt.get('dict_starttoken')
        self.end_token = opt.get('dict_endtoken')
        self.unk_token = opt.get('dict_unktoken')
        self.sep_token = opt.get('delimiter')
        if not self.load(opt.get('dict_file')):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.textfields = opt.get(
            'dict_textfields', DictionaryAgent.default_textfields
        ).split(",")

    def add_token(self, word):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def __getitem__(self, key):
        if type(key) == int:
            # return token from index, or unk_token
            return self.tokenizer.convert_ids_to_tokens([key])[0]
        elif type(key) == str:
            # return index from token, or unk_token's index, or None
            return self.tokenizer.convert_tokens_to_ids([key])[0]

    def __len__(self):
        return self.tokenizer.vocab_size

    def __setitem__(self, key, value):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def copy_dict(self, dictionary):
        raise NotImplementedError

    def max_freq(self):
        raise NotImplementedError

    def freqs(self):
        raise NotImplementedError

    def spacy_tokenize(self, text, **kwargs):
        raise NotImplementedError

    def spacy_span_tokenize(self, text):
        raise NotImplementedError

    def nltk_tokenize(self, text, building=False):
        raise NotImplementedError

    def gpt2_tokenize(self, text):
        raise NotImplementedError

    @staticmethod
    def re_tokenize(text):
        raise NotImplementedError

    @staticmethod
    def split_tokenize(text):
        raise NotImplementedError

    @staticmethod
    def space_tokenize(text):
        raise NotImplementedError

    def span_tokenize(self, text):
        raise NotImplementedError

    def tokenize(self, text, building=False):
        word_tokens = self.tokenizer.tokenize(text)
        return word_tokens

    def bpe_tokenize(self, text):
        raise NotImplementedError

    def add_to_dict(self, tokens):
        self.tokenizer.add_tokens(tokens)

    def remove_tail(self, min_freq):
        raise NotImplementedError

    def _remove_non_bpe(self):
        raise NotImplementedError

    def resize_to_max(self, maxtokens):
        raise NotImplementedError

    def load(self, filename):
        filepath = os.path.dirname(filename)
        print('Dictionary: loading dictionary from {}'.format(filepath))
        try:
            self.tokenizer = BertTokenizer.from_pretrained(filepath)
        except:
            return False
        return True

    def save(self, filename=None, append=False, sort=True):
        filepath = os.path.dirname(filename)
        print('Dictionary: saving dictionary to {}'.format(filepath))
        self.tokenizer.save_pretrained(filepath)
        self.tokenizer.save_vocabulary(filename)

    def sort(self, trim=True):
        raise NotImplementedError

    def parse(self, txt_or_vec, vec_type=list):
        raise NotImplementedError

    def txt2vec(self, text, vec_type=list):
        if vec_type == list or vec_type == tuple or vec_type == set:
            res = vec_type(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)))
        elif vec_type == np.ndarray:
            res = np.fromiter(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)), np.int)
        return res

    def vec2txt(self, vector, delimiter=' '):
        return self.tokenizer.decode(vector)

    def act(self):
        return super().act()

    def share(self):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError

