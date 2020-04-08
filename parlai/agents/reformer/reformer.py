# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Reformer Agents."""
from parlai.core.agents import Agent
from parlai.core.torch_generator_agent import TorchGeneratorAgent

from .modules import ReformerGeneratorModel


def add_common_cmdline_args(argparser):
    """Add common command line args."""
    argparser.add_argument(
        '-esz',
        '--embedding-size',
        type=int,
        default=300,
        help='Size of all embedding layers',
    )
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument(
        '-hid',
        '--hiddensize',
        type=int,
        default=300,
        help='Hidden size of the RNN layers',
    )
    argparser.add_argument(
        '--dropout', type=float, default=0.0, help='Dropout used in Vaswani 2017.'
    )
    argparser.add_argument(
        '--attention-dropout',
        type=float,
        default=0.0,
        help='Dropout used after attention softmax.',
    )
    argparser.add_argument(
        '--relu-dropout',
        type=float,
        default=0.0,
        help='Dropout used after ReLU. From tensor2tensor.',
    )
    argparser.add_argument(
        '--n-heads', type=int, default=2, help='Number of multihead attention heads'
    )
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument(
        '--n-positions',
        type=int,
        default=None,
        hidden=True,
        help='Number of positional embeddings to learn. Defaults '
        'to truncate or 1024 if not provided.',
    )
    argparser.add_argument(
        '--n-segments',
        type=int,
        default=0,
        help='The number of segments that support the model. '
        'If zero no segment and no langs_embedding.',
    )
    argparser.add_argument(
        '--variant',
        choices={'aiayn', 'xlm'},
        default='aiayn',
        help='Chooses locations of layer norms, etc.',
    )
    argparser.add_argument(
        '--activation',
        choices={'relu', 'gelu'},
        default='relu',
        help='Nonlinear activation to use. AIAYN uses relu, but '
        'more recent papers prefer gelu.',
    )
    argparser.add_argument(
        '--output-scaling',
        type=float,
        default=1.0,
        help='scale the output of every Reformer by this quantity.',
    )
    argparser.add_argument(
        '--share-word-embeddings',
        type='bool',
        default=True,
        help='Share word embeddings table for candidate and context'
        'in the memory network',
    )
    argparser.add_argument(
        '--rnn-class',
        choices={'rnn', 'gru', 'lstm'},
        default='gru',
        help='Which type of RNN to use.',
    )
    argparser.add_argument(
        '--rnn-layers',
        type=int,
        default=1,
        help='Number of RNN layers.',
    )
    argparser.add_argument(
        '--recurrence',
        choices={'rnn', 'all', 'none'},
        default=False,
        help='Which recurrence strategy to use.',
    )


class Reformer(Agent):
    """Placeholder Reformer Agent.

    Placeholder class, which just throws an error telling the user to specify
    whether they want the ranker or the generator.
    """

    def __init__(self, opt, shared=None):
        raise RuntimeError(
            "`--model Reformer` is not a valid choice. Please select either "
            "`--model Reformer/ranker` or `--model Reformer/generator"
        )


class ReformerGeneratorAgent(TorchGeneratorAgent):
    """ReformerGeneratorAgent.

    Implementation of TorchGeneratorAgent, where the model is a Reformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Reformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(ReformerGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """Build and return model."""
        model = ReformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model
