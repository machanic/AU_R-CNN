
import chainer.links as L
import chainer.functions as F
import chainer
from functools import partial
from chainer import Sequential
import numpy as np

def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)
    return tuple(padding_shape)

def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init

class Unit3Dpy(chainer.Chain):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn

        with self.init_scope():
            if padding == "SAME":
                padding_shape = get_padding_shape(kernel_size, stride)
                simplify_pad, pad_size = simplify_padding(padding_shape)
                self.simplify_pad = simplify_pad
            elif padding == "VALID":
                padding_shape = 0
            else:
                raise ValueError('padding should be in [VALID|SAME] but got {}'.format(padding))
            if padding == "SAME":
                if not self.simplify_pad:
                    self.conv3d = L.ConvolutionND(3, in_channels, out_channels, kernel_size, stride=stride, nobias=not use_bias)
                else:
                    self.conv3d = L.ConvolutionND(3, in_channels, out_channels, kernel_size, stride=stride, pad=pad_size,
                                                  nobias=not use_bias)
            elif padding == 'VALID':
                self.conv3d = L.ConvolutionND(3, in_channels, out_channels, kernel_size, pad=padding_shape, stride=stride,
                                              nobias=not use_bias)
            else:
                raise ValueError("padding should be in [VALID|SAME] but got {}".format(padding))

            if self.use_bn:
                self.batch3d = L.BatchNormalization(out_channels)  # FIXME BN3D?
            if activation == 'relu':
                self.activation = F.relu
            self.padding_shape = padding_shape

    def __call__(self, inp):
        if self.padding == "SAME" and self.simplify_pad is False:
            inp = F.pad(inp, pad_width=self.padding_shape, mode='constant', constant_values=0)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = self.activation(out)
        return out

class MaxPool3dTFPadding(chainer.Chain):

    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = partial(F.pad, pad_width=padding_shape,constant_values=0)
        self.pool = partial(F.max_pooling_nd, ksize=kernel_size, stride=stride)  # TODO note that the kernal_size must be 3D

    def __call__(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out

class Mixed(chainer.Chain):

    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()

        with self.init_scope():
            # Branch 0
            self.branch_0 = Unit3Dpy(in_channels, out_channels[0], kernel_size=(1,1,1))

            # Branch 1
            branch_1_conv1 = Unit3Dpy(
                in_channels, out_channels[1], kernel_size=(1, 1, 1))
            branch_1_conv2 = Unit3Dpy(
                out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
            self.branch_1 = Sequential(branch_1_conv1,branch_1_conv2)

            # Branch 2
            branch_2_conv1 = Unit3Dpy(
                in_channels, out_channels[3], kernel_size=(1, 1, 1))
            branch_2_conv2 = Unit3Dpy(
                out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
            self.branch_2 = Sequential(branch_2_conv1, branch_2_conv2)

            # Branch 3
            branch_3_pool = MaxPool3dTFPadding(
                kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
            branch_3_conv2 = Unit3Dpy(
                in_channels, out_channels[5], kernel_size=(1, 1, 1))
            self.branch_3 = Sequential(branch_3_pool, branch_3_conv2)

    def __call__(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = F.concat([out_0, out_1, out_2, out_3], axis=1)
        return out


class I3DFeatureExtractor(chainer.Chain):

    def __init__(self, modality='rgb'):
        super(I3DFeatureExtractor, self).__init__()
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        with self.init_scope():
            self.modality = modality
            conv3d_1a_7x7 = Unit3Dpy(
                out_channels=64,
                in_channels=in_channels,
                kernel_size=(7, 7, 7),
                stride=(2, 2, 2),
                padding='SAME')
            # 1st conv-pool
            self.conv3d_1a_7x7 = conv3d_1a_7x7
            self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(kernel_size=(1,3,3), stride=(1,2,2), padding='SAME')
            # conv conv
            conv3d_2b_1x1 = Unit3Dpy(
                out_channels=64,
                in_channels=64,
                kernel_size=(1, 1, 1),
                padding='SAME')
            self.conv3d_2b_1x1 = conv3d_2b_1x1
            conv3d_2c_3x3 = Unit3Dpy(
                out_channels=192,
                in_channels=64,
                kernel_size=(3, 3, 3),
                padding='SAME')
            self.conv3d_2c_3x3 = conv3d_2c_3x3
            self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
                kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

            # Mixed_3b
            self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
            self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])
            self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
                kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

            # Mixed 4
            self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
            self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
            self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
            self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
            self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

            self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
                kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

            # Mixed 5
            self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
            self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])



    def __call__(self, inp):
        #  B, C, T, H, W
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        return out   #  B, C, T, H, W


class I3DRoIHead(chainer.Chain):

    def __init__(self, out_channel, roi_size, spatial_scale,dropout_prob):
        super(I3DRoIHead, self).__init__()
        with self.init_scope():
            self.out_channel = out_channel
            self.avg_pool = partial(F.average_pooling_nd, ksize=(2, 7, 7), stride=(1, 1, 1))
            self.dropout = partial(F.dropout, ratio=dropout_prob)
            self.roi_size = roi_size
            self.spatial_scale = spatial_scale

            self.conv3d_0c_1x1 = Unit3Dpy(
                in_channels=1024,
                out_channels=out_channel,
                kernel_size=(1, 1, 1),
                activation=None,
                use_bias=True,
                use_bn=False)


    def __call__(self, x, rois, roi_indices, extract_layer="avg_pool"):
        # x is shape = (B, C, T, H, W)
        mini_batch_size, channel, seq_len, height, width = x.shape
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)  # None means np.newaxis, concat along column
        pool = self.roi_pooling_2d_yx(x, indices_and_rois, self.roi_size, self.roi_size, self.spatial_scale)  # B * F, C, T, H, W
        out = self.avg_pool(pool)  # B * F, C, T, H, W
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)  # B * F, C, T, H, W; where C = n_class
        out = F.squeeze(out, axis=3)  # B * F, C, T, H
        out = F.squeeze(out, axis=3)  # B * F, C, T
        assert out.ndim == 3
        out = F.reshape(out, shape=(mini_batch_size,pool.shape[0]/mini_batch_size, out.shape[1], out.shape[2] ))  # B, F, C, T
        out = F.transpose(out, axes=(0,3,1,2))  # B, T, F, C
        assert out.shape[-1] == self.out_channel
        out = F.reshape(out, shape=(-1, self.out_channel))  # B * T* F, 2048
        return out


    def roi_pooling_2d_yx(self, x, indices_and_rois, outh, outw, spatial_scale):  #  所以roi可以在T时间轴每一个图都有roi
        # x is shape = B, C, T, H ,W
        x = F.transpose(x, axes=(0, 2, 1, 3, 4)) # B, T, C, H, W
        mini_batch, seq_len, channel, height, width = x.shape
        x = F.reshape(x, shape=(mini_batch * seq_len, channel, height, width))

        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        box_mini_batch = xy_indices_and_rois.shape[0]
        # x is B * T, C, H, W
        pool = F.roi_pooling_2d(
            x, xy_indices_and_rois, outh, outw, spatial_scale)  # R, C, H, W, where R = B * T * F
        assert pool.shape[0] == box_mini_batch
        pool = F.reshape(pool, shape=(mini_batch, seq_len, box_mini_batch / (mini_batch * seq_len), channel,
                                      self.roi_size, self.roi_size)) # B, T, F, C, H, W
        pool = F.transpose(pool, axes=(0, 2, 3, 1, 4, 5))  # B, F,C, T, H, W
        pool = F.reshape(pool, shape=(box_mini_batch/ seq_len, channel, seq_len,
                                      self.roi_size, self.roi_size)) # B * F, C, T, H, W
        return pool


if __name__ == "__main__":
    i3d = I3DFeatureExtractor("rgb")
    weight = np.load("/home/machen/download2/model_rgb.npz")
    head = I3DRoIHead(400, 7, 1/16.0, dropout_prob=0.0)
    chainer.serializers.load_npz("/home/machen/download2/model_rgb.npz", head ,strict=True)

    for name in weight.files:
        print(name)
        if hasattr(i3d, name):
            getattr(i3d, name).copydata(weight[name])
            print("transfer")
    print("name ")