#!/usr/bin/env python3

# Copyright (c) Facebook, Inc., its affiliates and Shaojie Jiang.

from .module import TransformerModel

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.agents.seq2seq.modules import opt_to_kwargs


class PyTransformerAgent(TorchGeneratorAgent):
    """"""

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('PyTransformer Arguments')
        agent.add_argument(
            '-esz',
            '--embedding-size',
            type=int,
            default=300,
            help='Size of all embedding layers',
        )
        agent.add_argument('-nl', '--n-layers', type=int, default=2)
        agent.add_argument(
            '-hid',
            '--ffn-size',
            type=int,
            default=300,
            help='Hidden size of the FFN layers',
        )
        agent.add_argument(
            '--dropout', type=float, default=0.0, help='Dropout used in Vaswani 2017.'
        )
        agent.add_argument( # TODO: add support
            '--attention-dropout',
            type=float,
            default=0.0,
            help='Dropout used after attention softmax.',
        )
        agent.add_argument( # TODO: add support
            '--relu-dropout',
            type=float,
            default=0.0,
            help='Dropout used after ReLU. From tensor2tensor.',
        )
        agent.add_argument(
            '--n-heads', type=int, default=2, help='Number of multihead attention heads'
        )
        agent.add_argument('--learn-positional-embeddings', type='bool', default=False)
        agent.add_argument('--embeddings-scale', type='bool', default=True) # TODO: add support
        agent.add_argument(
            '--n-positions',
            type=int,
            default=None,
            hidden=True,
            help='Number of positional embeddings to learn. Defaults '
            'to truncate or 1024 if not provided.',
        )
        agent.add_argument( # TODO: add support
            '--n-segments',
            type=int,
            default=0,
            help='The number of segments that support the model. '
            'If zero no segment and no langs_embedding.',
        )
        agent.add_argument( # TODO: add support
            '--variant',
            choices={'aiayn', 'xlm'},
            default='aiayn',
            help='Chooses locations of layer norms, etc.',
        )
        agent.add_argument(
            '--activation',
            choices={'relu', 'gelu'},
            default='relu',
            help='Nonlinear activation to use. AIAYN uses relu, but '
            'more recent papers prefer gelu.',
        )
        agent.add_argument( # TODO: add support
            '--output-scaling',
            type=float,
            default=1.0,
            help='scale the output of every transformer by this quantity.',
        )
        agent.add_argument(
            '--share-lt',
            choices={'enc_dec', 'dec_out', 'unique', 'all'},
            default='unique',
            help='Share word embeddings table for candidate and context'
            'in the memory network',
        )

        super(PyTransformerAgent, cls).add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        return 0.1

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'PyTransformer'

    def build_model(self, states=None):
        opt = self.opt
        kwargs = opt_to_kwargs(opt)

        if not states:
            states = {}
        model = TransformerModel(
            len(self.dict),
            opt['embedding_size'],
            opt['ffn_size'],
            nlayers=opt['n_layers'],
            nheads=opt['n_heads'],
            npos=opt['n_positions'],
            dropout=opt['dropout'],
            lookuptable=opt['share_lt'],
            padding_idx=self.NULL_IDX,
            start_idx=self.START_IDX,
            end_idx=self.END_IDX,
            unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            activation=opt['activation'],
            learn_position=opt['learn_positional_embeddings'],
        )

        if opt.get('dict_tokenizer') == 'bpe' and opt['embedding_type'] != 'random':
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(model.decoder.lt.weight, opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(
                    model.encoder.lt.weight, opt['embedding_type'], log=False
                )

        if states:
            # set loaded states if applicable
            model.load_state_dict(states['model'])

        if opt['embedding_type'].endswith('fixed'):
            print('PyTransformer: fixing embedding weights.')
            model.decoder.lt.weight.requires_grad = False
            model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                model.output.weight.requires_grad = False
        return model
