from enum import Enum

import chainer


class ConvRNNType(Enum):
    conv_lstm = "conv_lstm"
    conv_qrnn  = "conv_qrnn"
    conv_rcnn = "conv_rcnn"
    sep_conv_lstm = "sep_conv_lstm"
    bn_conv_lstm = "bn_conv_lstm"
    fc_lstm = "fc_lstm"
    def __str__(self):
        return self.value



class TwoStreamMode(Enum):
    rgb_flow = "rgb_flow"
    rgb = "rgb"
    optical_flow = "optical_flow"

    def __str__(self):
        return self.value


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