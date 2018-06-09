
from functools import partial

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Sequential

from time_axis_rcnn.model.time_segment_network.dilated_convolution_1d import DilatedConvolution1D


class Chomp1d(chainer.Chain):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def __call__(self, x):
        return x[:, :, :-self.chomp_size]  # N, C, W


class TemporalBlock(chainer.Chain):
    def __init__(self, n_inputs, n_outputs, ksize, stride, dilate, pad, dropout=0.2):
        super(TemporalBlock, self).__init__()
        with self.init_scope():
            self.conv1 = DilatedConvolution1D(n_inputs, n_outputs, ksize, stride=stride, pad=pad, dilate=dilate)
            self.chomp1 = Chomp1d(pad)
            self.relu1 = F.relu
            self.dropout1 = partial(F.dropout, ratio=dropout)
            self.conv2 = DilatedConvolution1D(n_outputs, n_outputs, ksize, stride=stride, pad=pad, dilate=dilate)
            self.chomp2 = Chomp1d(pad)
            self.relu2 = F.relu
            self.dropout2 = partial(F.dropout, ratio=dropout)
            self.net = Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
            self.downsample = L.ConvolutionND(1, n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = F.relu

    def __call__(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(chainer.Chain):

    def __init__(self, num_inputs, num_channels, ksize=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.layers = []
        num_levels = len(num_channels)
        with self.init_scope():
            for i in range(num_levels):
                dilate = 2 ** i
                in_channels = num_inputs if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]
                setattr(self, "layer_{}".format(i), TemporalBlock(in_channels, out_channels, ksize,stride=1, dilate=dilate,
                                                                  pad=(ksize-1)* dilate, dropout=dropout))
                self.layers.append(getattr(self,"layer_{}".format(i)))
            self.network = Sequential(*self.layers)

    def __call__(self, x):
        return self.network(x)


class TcnBackbone(chainer.Chain):

    def __init__(self, conv_layer_num, in_channels, out_channels):
        channel_sizes = [in_channels] + [in_channels//2] * (conv_layer_num - 2) + [out_channels]
        super(TcnBackbone, self).__init__()
        with self.init_scope():
            self.tcn = TemporalConvNet(in_channels, channel_sizes, ksize=3)

    def __call__(self, x):
        h = self.tcn(x)
        assert not self.xp.isnan(self.xp.sum(h.data)), h.data
        return h