import numpy as np
from space_time_AU_rcnn.model.AU_rcnn.roi_tools.roi_align_2d import roi_align_2d
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import ResNet101Layers
from space_time_AU_rcnn.model.AU_rcnn.au_rcnn import AU_RCNN
import config
from chainer import initializers
from collections import namedtuple, OrderedDict
from chainer.serializers import npz


class AU_RCNN_MobilenetV1(AU_RCNN):

    """Faster R-CNN based on MobileNet.

    When you specify the path of a pre-trained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    For descriptions on the interface of this model, please refer to
    :class:`AU_rcnn.links.model.AU_rcnn.FasterRCNN`.

    :obj:`FasterRCNNVGG16` supports finer control on random initializations of
    weights by arguments
    :obj:`vgg_initialW`, :obj:`rpn_initialW`, :obj:`loc_initialW` and
    :obj:`score_initialW`.
    It accepts a callable that takes an array and edits its values.
    If :obj:`None` is passed as an initializer, the default initializer is
    used.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        pretrained_model (str): The destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/AU_rcnn/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.
        min_size (int): A preprocessing paramter for :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        vgg_initialW (callable): Initializer for the layers corresponding to
            the VGG-16 layers.
        rpn_initialW (callable): Initializer for Region Proposal Network
            layers.
        loc_initialW (callable): Initializer for the localization head.
        score_initialW (callable): Initializer for the score head.
        proposal_creator_params (dict): Key valued paramters for
            :obj:`AU_rcnn.links.model.AU_rcnn.ProposalCreator`.

    """



    feat_stride = 16

    def __init__(self,
                 pretrained_model_type=None,
                 min_size=512, max_size=512,
                 mean_file=None, classify_mode=False, n_class=0,use_roi_align=False
                 ):
        sizes = [128, 160, 192, 224]
        depth_mults = [0.75, 1.0]
        models = {}
        for size in sizes:
            for depth_mult in depth_mults:
                models[(depth_mult, size)] = "{0}/mobilenet_trained_model/chainer_mobilenet_v1_{1}_{2}.npz".format(
                    config.ROOT_PATH, depth_mult, size)



        extractor = MobileNet_v1_Base()
        head = MobileNetHead(
            roi_size=7, spatial_scale=1. / self.feat_stride, n_class=n_class, classify_mode=classify_mode,
            use_roi_align=use_roi_align
        )
        mean_array = np.load(mean_file)
        print("loading mean_file in: {} done".format(mean_file))

        if pretrained_model_type:
            pretrained_model_type = tuple(pretrained_model_type)
            extractor = MobileNet_v1_Base(depth_multiplier=pretrained_model_type[0])
            assert pretrained_model_type in models
            pretrained_path = models[pretrained_model_type]
            chainer.serializers.load_npz(pretrained_path, extractor)
            print("load pretrained mobilenet v1 file: {}".format(pretrained_path))

        super(AU_RCNN_MobilenetV1, self).__init__(
            extractor,
            head,
            mean=mean_array,
            min_size=min_size,
            max_size=max_size
        )


class MobileNetHead(chainer.Chain):

    """Faster R-CNN Head for VGG-16 based implementation.

    This class is used as a head for Faster R-CNN.
    This outputs class-wise classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        vgg_initialW (callable): Initializer for the layers corresponding to
            the VGG-16 layers.
        score_initialW (callable): Initializer for the score head.

    """

    def __init__(self, roi_size=7, spatial_scale=1/16.0, n_class=22, classify_mode=False,use_roi_align=False):
        # n_class includes the background
        super(MobileNetHead, self).__init__()
        self.classify_mode = classify_mode
        self.use_roi_align = use_roi_align
        initialW = initializers.HeNormal()
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale  # 这个很关键,一般都是1/16.0
        with self.init_scope():
            self.convert_feature_dim_fc = L.Linear(75264, 2048)
            self.score = L.Linear(2048, n_class)

    def __call__(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (~chainer.Variable): 4D image variable.
            rois (array): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (array): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # x shape = B, 1536, 32, 32
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)  # None means np.newaxis, concat along column
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale,self.use_roi_align)
        h = pool  # shape = R', 1536, 7, 7
        h = F.reshape(h, shape=(h.shape[0], 75264))
        h = self.convert_feature_dim_fc(h)  # R', 2048
        if self.classify_mode:
            h = self.score(F.relu(h))
        return h


class MobileNet_v1_Base(chainer.Chain):

    def __init__(self, final_endpoint='Conv2d_13_pointwise_batchnorm', min_depth=8, depth_multiplier=1.0,
                 output_stride=None, use_explicit_padding=False):
        super(MobileNet_v1_Base, self).__init__()
        initialW = initializers.HeNormal()
        self.depth = lambda d: max(int(d * depth_multiplier), min_depth)
        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')

        if output_stride is not None and output_stride not in [8, 16, 32]:
            raise ValueError('Only allowed output_stride values are 8, 16, 32.')
        Conv = namedtuple('Conv', ['kernel', 'stride','inchannel', 'depth'])
        DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'inchannel', 'stride', 'depth'])
        self.conv_defs = [
                Conv(kernel=[3, 3], stride=2, inchannel=3, depth=32),
                DepthSepConv(kernel=[3, 3], stride=1, inchannel=32, depth=64),
                DepthSepConv(kernel=[3, 3], stride=2, inchannel=64, depth=128),
                DepthSepConv(kernel=[3, 3], stride=1, inchannel=128,depth=128),
                DepthSepConv(kernel=[3, 3], stride=2, inchannel=128, depth=256),
                DepthSepConv(kernel=[3, 3], stride=1, inchannel=256,depth=256),
                DepthSepConv(kernel=[3, 3], stride=2, inchannel=256,depth=512),
                DepthSepConv(kernel=[3, 3], stride=1, inchannel=512,depth=512),
                DepthSepConv(kernel=[3, 3], stride=1, inchannel=512,depth=512),
                DepthSepConv(kernel=[3, 3], stride=1, inchannel=512,depth=512),
                DepthSepConv(kernel=[3, 3], stride=1, inchannel=512,depth=512),
                DepthSepConv(kernel=[3, 3], stride=1, inchannel=512,depth=512),
                DepthSepConv(kernel=[3, 3], stride=2, inchannel=512, depth=1024),
                DepthSepConv(kernel=[3, 3], stride=1, inchannel=1024, depth=1024),
            ]
        self.scale_dict = {"Conv2d_11_pointwise_batchnorm" : "16_scale", "Conv2d_13_pointwise_batchnorm": "32_scale"}

        self.padding = 'SAME'
        pad = 1  # 3 x 3 conv kernal, with 1x1 padding
        if use_explicit_padding:
            self.padding = 'VALID'
            pad = 0
        with self.init_scope():
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            self.layer_names = []
            current_stride = 1
            rate = 1
            for i, conv_def in enumerate(self.conv_defs):
                end_point_base = 'Conv2d_{}'.format(i)
                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    # FIXME chainer的depthwise conv 不支持atrous rate or dilated rate
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride
                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    if use_explicit_padding:
                        self.layer_names.append(("use_explicit_padding", conv_def.kernel))
                    setattr(self, end_point, L.Convolution2D(in_channels=conv_def.inchannel, out_channels=self.depth(conv_def.depth),
                                                             ksize=conv_def.kernel, stride=conv_def.stride,pad=pad, nobias=False,
                                                             initialW=initialW))
                    self.layer_names.append(end_point)
                    bn_end_point = end_point + "_batchnorm"
                    setattr(self, bn_end_point, L.BatchNormalization(size=self.depth(conv_def.depth)))
                    self.layer_names.append(bn_end_point)
                    if end_point == final_endpoint:
                        return
                elif isinstance(conv_def, DepthSepConv):

                    end_point = end_point_base + '_depthwise'
                    if use_explicit_padding:
                        self.layer_names.append(("use_explicit_padding", conv_def.kernel, layer_rate))
                    setattr(self, end_point, L.DepthwiseConvolution2D(in_channels=self.depth(conv_def.inchannel), channel_multiplier=1,
                                                                      ksize=conv_def.kernel, stride=layer_stride, pad=pad,nobias=False,
                                                                      initialW=initialW))
                    self.layer_names.append(end_point)
                    bn_end_point = end_point + "_batchnorm"
                    setattr(self, bn_end_point, L.BatchNormalization(size=self.depth(conv_def.inchannel)))
                    self.layer_names.append(bn_end_point)
                    if end_point == final_endpoint:
                        return
                    end_point = end_point_base + '_pointwise'

                    setattr(self, end_point, L.Convolution2D(in_channels=self.depth(conv_def.inchannel), out_channels=self.depth(conv_def.depth),
                                                             ksize=[1,1], stride=1, pad=0, nobias=False, initialW=initialW))
                    self.layer_names.append(end_point)
                    bn_end_point = end_point + "_batchnorm"
                    setattr(self, bn_end_point, L.BatchNormalization(size=self.depth(conv_def.depth)))
                    self.layer_names.append(bn_end_point)
                    if end_point == final_endpoint:
                        return
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))

    def __call__(self, x): # x shape = (B, C, H, W), extract_layers = [Conv2d_13_pointwise, Conv2d_11_pointwise]
        extract_dict = {}
        for layer_name in self.layer_names:
            if isinstance(layer_name, tuple):
                if len(layer_name) == 3:
                    _, kernel, layer_rate = layer_name
                    x = self._fixed_padding(x, kernel, layer_rate)
                elif len(layer_name) == 2:
                    _, kernel = layer_name
                    x = self._fixed_padding(x, kernel)
            elif isinstance(layer_name, str):
                x = getattr(self, layer_name)(x)
            if layer_name in self.scale_dict:
                extract_dict[self.scale_dict[layer_name]] = x
        scale_32 = extract_dict["32_scale"]
        scale_16 = extract_dict["16_scale"]
        x = F.concat((F.resize_images(scale_32, output_shape=scale_16.shape[2:4]), scale_16), axis=1)
        extract_dict.clear()
        return x


    @classmethod
    def _transfer_each_depthwise_layer(cls, src, dst, layer_no, layer_type):
        getattr(dst, "Conv2d_{0}_{1}".format(layer_no, layer_type)).W.data[:] = \
            np.transpose(src['MobilenetV1/Conv2d_{0}_{1}/{2}:0'.format(layer_no, layer_type, "depthwise_weights"
            if layer_type == "depthwise" else "weights")], (3, 2, 0, 1))
        getattr(dst, "Conv2d_{0}_{1}_batchnorm".format(layer_no, layer_type)).avg_mean[:] =\
            src['MobilenetV1/Conv2d_{0}_{1}/BatchNorm/moving_mean:0'.format(layer_no, layer_type)]
        getattr(dst, "Conv2d_{0}_{1}_batchnorm".format(layer_no, layer_type)).avg_var[:] = \
            src['MobilenetV1/Conv2d_{0}_{1}/BatchNorm/moving_variance:0'.format(layer_no, layer_type)]
        getattr(dst, "Conv2d_{0}_{1}_batchnorm".format(layer_no, layer_type)).gamma.data[:] = \
            src['MobilenetV1/Conv2d_{0}_{1}/BatchNorm/gamma:0'.format(layer_no, layer_type)]
        getattr(dst, "Conv2d_{0}_{1}_batchnorm".format(layer_no, layer_type)).beta.data[:] = \
            src['MobilenetV1/Conv2d_{0}_{1}/BatchNorm/beta:0'.format(layer_no, layer_type)]



    @classmethod
    def _transfer_mobile_net_v1(cls, src, dst):
        # tf conv2D w shape = [filter_height, filter_width, in_channels, out_channels]
        # chainer conv2D w shape = [out_channel, inchannel, filter_height, filter_width]

        # tf depthwise conv2D W shape = [filter_height, filter_width, in_channels, 1]
        # chainer depthwise conv2D W shape =  [1, in_channels, filter_height, filter_width]

        dst.Conv2d_0.W.data[:] = np.transpose(src['MobilenetV1/Conv2d_0/weights:0'], (3,2,0,1))
        dst.Conv2d_0_batchnorm.avg_mean[:] = src['MobilenetV1/Conv2d_0/BatchNorm/moving_mean:0']
        dst.Conv2d_0_batchnorm.avg_var[:] = src['MobilenetV1/Conv2d_0/BatchNorm/moving_variance:0']
        dst.Conv2d_0_batchnorm.gamma.data[:] = src['MobilenetV1/Conv2d_0/BatchNorm/gamma:0']
        dst.Conv2d_0_batchnorm.beta.data[:] = src['MobilenetV1/Conv2d_0/BatchNorm/beta:0']

        for layer_no in range(1, 14):
            cls._transfer_each_depthwise_layer(src, dst, layer_no, "depthwise")
            cls._transfer_each_depthwise_layer(src, dst, layer_no, "pointwise")



    @classmethod
    def convert_tf_to_chainer(cls, tf_npz_path, chainer_npz_path, depth_mult):
        tf_model = np.load(tf_npz_path)
        chainermodel = cls(depth_multiplier=depth_mult)
        cls._transfer_mobile_net_v1(tf_model, chainermodel)
        npz.save_npz(chainer_npz_path, chainermodel, compression=False)

    def _fixed_padding(self, inputs, kernel_size, rate=1):
        '''
        ""Pads the input along the spatial dimensions independently of input size.
          Pads the input such that if it was used in a convolution with 'VALID' padding,
          the output would have the same dimensions as if the unpadded input was used
          in a convolution with 'SAME' padding.
          Args:
            inputs: A tensor of size [batch, height_in, width_in, channels].
            kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
            rate: An integer, rate for atrous convolution.
          Returns:
            output: A tensor of size [batch, height_out, width_out, channels] with the
              input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
        '''
        kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                                 kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
        pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
        pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
        pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
        padded_inputs = F.pad(inputs,  [[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]], [0, 0]],
                              mode='constant')
        return padded_inputs



def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale, use_roi_align):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    if use_roi_align:
        pool = roi_align_2d(x, xy_indices_and_rois, outh, outw, spatial_scale)
    else:
        pool = F.roi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool

if __name__ == "__main__":

    source_dir = "G:/Facial AU detection dataset/mobilenet pretrained model file/tensorflow_npz_file"
    target_dir = "G:/Facial AU detection dataset/mobilenet pretrained model file/chainer_npz_file"
    import os
    for tf_path in os.listdir(source_dir):
        depth_mult = 0.75 if "0.75" in tf_path else 1
        abs_tf_path  = source_dir + "/" + tf_path
        MobileNet_v1_Base.convert_tf_to_chainer(abs_tf_path,
                                                "{0}/{1}".format(target_dir, os.path.basename(abs_tf_path)), depth_mult)
