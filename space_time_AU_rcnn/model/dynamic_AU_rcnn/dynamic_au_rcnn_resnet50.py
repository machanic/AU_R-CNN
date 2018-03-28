import collections
import numpy as np
from chainer.links import ResNet50Layers

from space_time_AU_rcnn.model.AU_rcnn.roi_tools.roi_align_2d import roi_align_2d
import chainer
import chainer.functions as F
import chainer.links as L
import functools
from space_time_AU_rcnn.model.AU_rcnn.au_rcnn import AU_RCNN
import config
from chainer import initializers
from space_time_AU_rcnn.model.roi_space_time_net.conv_lstm.conv_lstm_block import ConvSRUCell

class Dynamic_AU_RCNN_Resnet50(AU_RCNN):

    """Faster R-CNN based on ResNet101.

    When you specify the path of a pre-trained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`voc07`: Loads weights trained with the trainval split of \
        PASCAL VOC2007 Detection Dataset.
    * :obj:`imagenet`: Loads weights trained with ImageNet Classfication \
        task for the feature extractor and the head modules. \
        Weights that do not have a corresponding layer in VGG-16 \
        will be randomly initialized.

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

    _models = {
        'voc07': {
            'n_fg_class': 20,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.3/faster_rcnn_vgg16_voc07_2017_06_06.npz'
        },
        'imagenet': {
            'path': "{}/caffe_model/VGG_ILSVRC_16_layers.npz".format(config.ROOT_PATH)
            # 'url': "http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel"
        },
        'resnet101': {
            'path': config.ROOT_PATH + '/caffe_model/ResNet-101-model.npz'
        }
    }

    feat_stride = 16

    def __init__(self,database,
                 pretrained_model=None,
                 min_size=512, max_size=512,
                 mean_file=None, n_class=12, classify_mode=False,use_roi_align=False,use_feature_map=False
                 ):

        extractor = ResnetFeatureExtractor()
        head = ResRoIHead(database=database,
            roi_size=14, spatial_scale=1. / self.feat_stride,
            n_class=n_class, classify_mode=classify_mode, use_roi_align=use_roi_align, use_feature_map=use_feature_map # 1/ 16.0 means after extract feature map, the map become 1/16 of original image, ROI bbox also needs shrink
        )
        mean_array = np.load(mean_file)
        print("loading mean_file in: {} done".format(mean_file))
        super(Dynamic_AU_RCNN_Resnet50, self).__init__(
            extractor,
            head,
            mean=mean_array,
            min_size=min_size,
            max_size=max_size
        )

        # if pretrained_model == 'resnet101':  # 只会走到这一elif里
        #     self._copy_imagenet_pretrained_resnet101(path=self._models['resnet101']['path'])
        #     print("load pretrained file: {} done".format(self._models['resnet101']['path']))
        # elif pretrained_model.endswith(".npz"):
        #     print("loading :{} to AU R-CNN ResNet-101".format(pretrained_model))
        #     chainer.serializers.load_npz(pretrained_model, self)

    def _copy_imagenet_pretrained_resnet50(self, path):
        pretrained_model = ResNet50Layers(pretrained_model=path)
        self.extractor.conv1.copyparams(pretrained_model.conv1)
        self.extractor.bn1.copyparams(pretrained_model.bn1)
        self.extractor.res2.copyparams(pretrained_model.res2)
        self.extractor.res3.copyparams(pretrained_model.res3)
        self.extractor.res4.copyparams(pretrained_model.res4)
        self.head.res5.copyparams(pretrained_model.res5)

class DynamicConvolution2D(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None):
        super(DynamicConvolution2D, self).__init__()
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        self.stride = stride
        with self.init_scope():
            self.cell = ConvSRUCell( input_dim=in_channels, hidden_dim=hidden_channels,
                                          kernel_size=ksize)
            if hidden_channels != out_channels:
                self.conv = L.Convolution2D(in_channels=hidden_channels, out_channels=out_channels,
                                                      ksize=ksize, stride=stride,pad=pad,nobias=nobias,
                                                      initialW=initialW, initial_bias=initial_bias)
    def __call__(self, input_tensor):
        """
            Parameters
            ----------
            input_tensor: todo
                5-D Tensor either of shape (b, t, c, h, w)
            Returns
            -------
            b, t, c, h, w
        """

        hidden_state = self._init_hidden(batch_size=input_tensor.shape[0], height=input_tensor.shape[-2],
                                         width=input_tensor.shape[-1])
        c = hidden_state

        all_h = self.cell(input_tensor=input_tensor, zero_state=c)  # B, T, C, H, W
        mini_batch, seq_len, channel, height, width = all_h.shape
        output = F.reshape(all_h, (mini_batch*seq_len, -1, height, width))
        if hasattr(self, "conv"):
            output = self.conv(output)
        output = F.reshape(output, (mini_batch, seq_len, output.shape[-3], output.shape[-2], output.shape[-1]))
        return output

    def _init_hidden(self, batch_size, height, width):
        return self.cell.init_hidden(batch_size, height, width)



class DynamicBatchNormalization(L.BatchNormalization):
    def __init__(self, size):
        super(DynamicBatchNormalization, self).__init__(size)

    def __call__(self, x,  **kwargs):
        # x is B, T, C, H, W
        mini_batch, seq_len, channel, height, width = x.shape
        x = x.reshape(mini_batch * seq_len, channel, height, width)
        x = super(DynamicBatchNormalization, self).__call__(x, **kwargs)
        return x.reshape(mini_batch, seq_len, channel, height, width)


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2, use_dynamic=False):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()
        self.use_dynamic = use_dynamic
        with self.init_scope():
            if not use_dynamic:
                self.conv1 = L.Convolution2D(
                    in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
                self.bn1 = L.BatchNormalization(ch)
                self.conv2 = L.Convolution2D(
                    ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
                self.bn2 = L.BatchNormalization(ch)
                self.conv3 = L.Convolution2D(
                    ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
                self.bn3 = L.BatchNormalization(out_size)

                self.conv4 = L.Convolution2D(
                    in_size, out_size, 1, stride, 0,
                    initialW=initialW, nobias=True)  # note that residule link has stride = 2
                self.bn4 = L.BatchNormalization(out_size)
            else:
                self.conv1 = DynamicConvolution2D(in_size, ch, in_size, 1, stride, 0, initialW=initialW,
                                                  nobias=True)
                self.bn1 = DynamicBatchNormalization(ch)
                self.conv2 = DynamicConvolution2D( ch, ch, ch, 3, 1, 1, initialW=initialW,
                                                  nobias=True)
                self.bn2 = DynamicBatchNormalization(ch)
                self.conv3 = DynamicConvolution2D(ch, out_size, ch, 1,1,0,
                                                  initialW=initialW,nobias=True)
                self.bn3 = DynamicBatchNormalization(out_size)
                self.conv4 = DynamicConvolution2D(in_size, out_size, in_size, 1, stride, 0,
                                                  initialW=initialW,
                                                  nobias=True)
                self.bn4 = DynamicBatchNormalization(out_size)

    def __call__(self, x):

        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)





class BottleNeckB(chainer.Chain):

    def __init__(self,  in_size, ch, use_dynamic=False):
        super(BottleNeckB, self).__init__()
        self.use_dynamic = use_dynamic
        initialW = initializers.HeNormal()

        with self.init_scope():
            if not self.use_dynamic:
                self.conv1 = L.Convolution2D(
                    in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
                self.bn1 = L.BatchNormalization(ch)
                self.conv2 = L.Convolution2D(
                    ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
                self.bn2 = L.BatchNormalization(ch)
                self.conv3 = L.Convolution2D(
                    ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
                self.bn3 = L.BatchNormalization(in_size)
            else:
                self.conv1 = DynamicConvolution2D(in_size, ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
                self.bn1 = DynamicBatchNormalization(ch)
                self.conv2 = DynamicConvolution2D(ch, ch, ch, 3,1,1, initialW=initialW, nobias=True)
                self.bn2 = DynamicBatchNormalization(ch)
                self.conv3 = DynamicConvolution2D(ch, in_size, ch, 1,1,0, initialW=initialW, nobias=True)
                self.bn3 = DynamicBatchNormalization(in_size)


    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)

class Block(chainer.Chain):

    def __init__(self,  layer, in_size, ch, out_size, stride=2, use_dynamic=True):
        super(Block, self).__init__()
        self.add_link('a', BottleNeckA( in_size, ch, out_size, stride, use_dynamic))
        for i in range(1, layer):
            self.add_link('b{}'.format(i), BottleNeckB(out_size, ch, use_dynamic))
        self.layer = layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.layer):
            h = self['b{}'.format(i)](h)

        return h


class ResRoIHead(chainer.Chain):

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

    def __init__(self, database, roi_size, spatial_scale, n_class, classify_mode=False, use_roi_align=False,
                 use_feature_map=False):
        # n_class includes the background
        super(ResRoIHead, self).__init__()
        self.database = database
        self.use_roi_align = use_roi_align
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale  # 这个很关键,一般都是1/16.0
        self.activation = dict()
        self.classify_mode = classify_mode
        self.use_feature_map = use_feature_map
        with self.init_scope():
            self.res5 = Block(3, 1024, 512, 2048, use_dynamic=True)
            self.functions = collections.OrderedDict([
                ('res5',  [self.res5]),
            ])
            if (not use_feature_map) :
                self.functions["avg_pool"] = [functools.partial(F.average_pooling_2d, ksize=7, stride=1)]
            if classify_mode:
                self.score = L.Linear(2048, n_class)
                self.functions["reshape"] = [functools.partial(F.reshape, shape=(-1, 2048))]
                self.functions["score"] = [self.score]


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
        mini_batch, seq_len, channel, height, width = x.shape
        x = x.reshape(mini_batch * seq_len, channel, height, width)
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)  # None means np.newaxis, concat along column
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, self.use_roi_align)  # R', C, H, W. where R' is batch_index = all box number
        pool = pool.reshape(mini_batch, seq_len, config.BOX_NUM[self.database], pool.shape[-3], pool.shape[-2], pool.shape[-1])
        h = F.transpose(pool, axes=(0,2,1,3,4,5,))  # B, F, T, C, H, W
        h = F.reshape(h, shape=(h.shape[0] * h.shape[1], h.shape[2], h.shape[3], h.shape[4], h.shape[5]))  # B*F, T, C, H, W
        for key, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
            if key == "res5":
                h =h.reshape(h.shape[0]*h.shape[1], h.shape[2], h.shape[3], h.shape[4]) # B*F*T, C, H, W

        if self.use_feature_map:
            scale_16 = pool  # B, T, F, C, H, W
            scale_32 = h     # B*F*T, C, H, W
            scale_32 = scale_32.reshape(mini_batch, config.BOX_NUM[self.database], seq_len, scale_32.shape[-3],
                                        scale_32.shape[-2], scale_32.shape[-1])  # B, F, T, C, H, W
            scale_32 = F.transpose(scale_32, (0, 2, 1,3,4,5)) # B, T, F, C, H, W
            scale_16 = scale_16.reshape(scale_16.shape[0] * scale_16.shape[1] * scale_16.shape[2], scale_16.shape[3],
                                        scale_16.shape[4], scale_16.shape[5])
            scale_32 = scale_32.reshape(scale_32.shape[0] * scale_32.shape[1] * scale_32.shape[2], scale_32.shape[3],
                                        scale_32.shape[4], scale_32.shape[5])
            mix_output = F.concat((F.resize_images(scale_32, output_shape=scale_16.shape[2:4]), scale_16), axis=1)
            return mix_output  # B*T*F, C, H, W
        else:
            h = h.reshape(mini_batch, config.BOX_NUM[self.database], seq_len, -1)
            h = F.transpose(h, axes=(0, 2, 1, 3)) # B, T, F, 12
        return h


class ResnetFeatureExtractor(chainer.Chain):

    """Truncated VGG-16 that extracts a conv5_3 feature map.

    Args:
        initialW (callable): Initializer for the weights.

    """

    def __init__(self):

        super(ResnetFeatureExtractor, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3, initialW=initializers.HeNormal(), nobias=True)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(3, 64, 64, 256, 1, use_dynamic=False)
            self.res3 = Block(4, 256, 128, 512, use_dynamic=False)
            self.res4 = Block(6, 512, 256, 1024, use_dynamic=False)

    def __call__(self, x):  # B, T, C, H, W
        mini_batch, seq_len, channel, height, width = x.shape
        x = x.reshape(mini_batch * seq_len, channel, height, width)
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = h.reshape(mini_batch, seq_len, h.shape[-3], h.shape[-2], h.shape[-1])
        return h


def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale, use_roi_align):

    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    if use_roi_align:
        pool = roi_align_2d(x, xy_indices_and_rois, outh, outw, spatial_scale)
    else:
        pool = F.roi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)
