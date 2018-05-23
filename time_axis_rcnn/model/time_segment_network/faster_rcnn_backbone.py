from collections import defaultdict

import chainer
import chainer.functions as F
import config
from time_axis_rcnn.model.time_segment_network.dilated_convolution_1d import DilatedConvolution1D


class FasterBackbone(chainer.Chain):

    def __init__(self, database, conv_layer_num, in_channels, out_channels):
        super(FasterBackbone, self).__init__()
        dilation_rates = [2, 3, 5, 7, 9]
        mid_channels = in_channels // 2
        self.conv_layers = defaultdict(list)
        self.groups = config.BOX_NUM[database]
        with self.init_scope():
            for i in range(conv_layer_num):
                if i != 0:
                    in_channels = mid_channels
                if i == conv_layer_num - 1:
                    mid_channels = out_channels
                for group_id in range(self.groups):
                    setattr(self, "conv_#{0}_layer_{1}".format(group_id, i),DilatedConvolution1D(in_channels, mid_channels,
                                                                        ksize=3, stride=1,
                                                                         dilate=dilation_rates[i%len(dilation_rates)],
                                                                        nobias=True))  # Note That we use one group conv
                    self.conv_layers[group_id].append("conv_#{0}_layer_{1}".format(group_id, i))


    def __call__(self, x, AU_group_id_array, seq_len):
        output_list = []
        for batch_idx, group_id in enumerate(AU_group_id_array):
            x_inside_batch = F.expand_dims(x[batch_idx], axis=0)  # 1,C,W
            for conv_layer in self.conv_layers[group_id]:
                x_inside_batch = getattr(self, conv_layer)(x_inside_batch)
            output_list.append(x_inside_batch)
            assert not self.xp.isnan(self.xp.sum(x_inside_batch.data))
        return F.concat(output_list, axis=0)  # B, C, W