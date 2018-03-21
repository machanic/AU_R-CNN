from enum import Enum

import chainer


class NoneModule(chainer.Chain):
    def __init__(self, n_layers, insize, outsize, initialW=None, use_bi_lstm=False):
        super(NoneModule, self).__init__()
        pass


class TemporalEdgeMode(Enum):
    rnn = 'rnn'
    ld_rnn = 'ld_rnn'
    bi_rnn = 'bi_rnn'
    bi_ld_rnn = 'bi_ld_rnn'
    no_temporal = 'no_temporal'  # only use/or even not use spatial edge; use 1D conv layer to simulate
    def __str__(self):
        return self.value


class NeighborMode(Enum):
    concat_all = 'concat_all'
    attention_fuse = 'attention_fuse'
    random_neighbor = 'random_neighbor'
    no_neighbor = 'no_neighbor'  # only self node, do not want neighbor
    def __str__(self):
        return self.value


class SpatialEdgeMode(Enum):
    rnn = 'rnn'
    ld_rnn = 'ld_rnn'
    bi_rnn = 'bi_rnn'
    bi_ld_rnn = 'bi_ld_rnn'
    no_edge = 'no_edge'
    def __str__(self):
        return self.value


class SpatialSequenceType(Enum):
    cross_time = 'cross_time'
    in_frame = "in_frame"
    def __str__(self):
        return self.value