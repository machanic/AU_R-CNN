import chainer
import chainer.links as L
import chainer.functions as F
from time_axis_rcnn.model.time_segment_network.util.links.weight_normalization import convert_with_weight_normalization as weight_norm


class DilatedConvolution1D(chainer.Chain):

    def __init__(self,in_channels, out_channels, ksize=None, stride=1, dilate=1, pad=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(DilatedConvolution1D, self).__init__()
        self.out_channels = out_channels
        self.ksize= ksize
        # self.pad = ((ksize - 1) * (dilate - 1) + ksize)//2
        with self.init_scope():
            self.conv2D = weight_norm(L.Convolution2D, in_channels, out_channels, ksize=(ksize, 1),
                                          stride=(stride,1), pad=(pad, 0),
                                          dilate=(dilate, 1), groups=1, nobias=nobias,
                                          initialW=initialW, initial_bias=initial_bias)


    def __call__(self, x):  # x is shape of (N,C,W)
        # N is set to number of AU groups (bounding box number)
        assert x.ndim == 3
        N = x.shape[0]
        W = x.shape[2]
        x = F.reshape(x, shape=(x.shape[0],  x.shape[1], x.shape[2], 1))  # FIXME , width 长度有变化 conv2D ? 472
        x = self.conv2D(x)  # N, C,W,1
        x = F.squeeze(x, axis=-1)
        return x


