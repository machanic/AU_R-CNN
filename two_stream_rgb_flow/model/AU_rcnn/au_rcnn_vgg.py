import collections
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import VGG16Layers
from lstm_end_to_end.model.AU_rcnn.au_rcnn import AU_RCNN

from lstm_end_to_end.utils import download_model
from lstm_end_to_end.model.AU_rcnn.roi_tools.roi_align_2d import roi_align_2d
import config


class AU_RCNN_VGG16(AU_RCNN):

    """Faster R-CNN based on VGG-16.

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
        'vgg_face': {
            'path': '{}/caffe_model/vgg_face.npz'.format(config.ROOT_PATH),
        }
    }
    feat_stride = 16

    def __init__(self,
                 pretrained_model=None,
                 min_size=512, max_size=512,
                 vgg_initialW=None,
                 mean_file=None, use_roi_align=False
                 ):

        if vgg_initialW is None and pretrained_model:
            vgg_initialW = chainer.initializers.constant.Zero()

        extractor = VGG16FeatureExtractor(initialW=vgg_initialW)
        head = VGG16RoIHead(
            roi_size=7, spatial_scale=1. / self.feat_stride, # 1/ 16.0 means after extract feature map, the map become 1/16 of original image, ROI bbox also needs shrink
            vgg_initialW=vgg_initialW,use_roi_align=use_roi_align
        )

        mean_array = np.load(mean_file)

        print("loading mean_file in: {} done".format(mean_file))
        super(AU_RCNN_VGG16, self).__init__(
            extractor,
            head,
            mean=mean_array,
            min_size=min_size,
            max_size=max_size
        )

        if pretrained_model in self._models and 'url' in self._models[pretrained_model]:
            path = download_model(self._models[pretrained_model]['url'])
            chainer.serializers.load_npz(path, self)
        elif pretrained_model == 'imagenet':  # 只会走到这一elif里
            print("loading:{} imagenet pretrained model".format(self._models['imagenet']['path']))
            model_path = self._models['imagenet']['path']
            if model_path.endswith(".npz"):
                self._copy_imagenet_pretrained_vgg16(path=model_path)

        elif pretrained_model.endswith("npz"):
            print("loading :{} to AU R-CNN VGG16".format(pretrained_model))
            chainer.serializers.load_npz(pretrained_model, self)  #FIXME 我修改了最后加了一层fc8， 变成1024维向量，但是无法load


    def _transfer_vgg(self, src, dst):
        dst.conv1_1.W.data[:] = src["conv1_1"].W.data
        dst.conv1_1.b.data[:] = src["conv1_1"].b.data
        dst.conv1_2.W.data[:] = src["conv1_2"].W.data
        dst.conv1_2.b.data[:] = src["conv1_2"].b.data
        dst.conv2_1.W.data[:] = src["conv2_1"].W.data
        dst.conv2_1.b.data[:] = src["conv2_1"].b.data
        dst.conv2_2.W.data[:] = src["conv2_2"].W.data
        dst.conv2_2.b.data[:] = src["conv2_2"].b.data
        dst.conv3_1.W.data[:] = src["conv3_1"].W.data
        dst.conv3_1.b.data[:] = src["conv3_1"].b.data
        dst.conv3_2.W.data[:] = src["conv3_2"].W.data
        dst.conv3_2.b.data[:] = src["conv3_2"].b.data
        dst.conv3_3.W.data[:] = src["conv3_3"].W.data
        dst.conv3_3.b.data[:] = src["conv3_3"].b.data
        dst.conv4_1.W.data[:] = src["conv4_1"].W.data
        dst.conv4_1.b.data[:] = src["conv4_1"].b.data
        dst.conv4_2.W.data[:] = src["conv4_2"].W.data
        dst.conv4_2.b.data[:] = src["conv4_2"].b.data
        dst.conv4_3.W.data[:] = src["conv4_3"].W.data
        dst.conv4_3.b.data[:] = src["conv4_3"].b.data
        dst.conv5_1.W.data[:] = src["conv5_1"].W.data
        dst.conv5_1.b.data[:] = src["conv5_1"].b.data
        dst.conv5_2.W.data[:] = src["conv5_2"].W.data
        dst.conv5_2.b.data[:] = src["conv5_2"].b.data
        dst.conv5_3.W.data[:] = src["conv5_3"].W.data
        dst.conv5_3.b.data[:] = src["conv5_3"].b.data
        dst.fc6.W.data[:] = src["fc6"].W.data
        dst.fc6.b.data[:] = src["fc6"].b.data
        # dst.fc7.W.data[:] = src["fc7"].W.data
        # dst.fc7.b.data[:] = src["fc7"].b.data

    def _copy_imagenet_pretrained_vgg16(self, path):
        pretrained_model = VGG16Layers(pretrained_model=path)
        self.extractor.conv1_1.copyparams(pretrained_model.conv1_1)
        self.extractor.conv1_2.copyparams(pretrained_model.conv1_2)
        self.extractor.conv2_1.copyparams(pretrained_model.conv2_1)
        self.extractor.conv2_2.copyparams(pretrained_model.conv2_2)
        self.extractor.conv3_1.copyparams(pretrained_model.conv3_1)
        self.extractor.conv3_2.copyparams(pretrained_model.conv3_2)
        self.extractor.conv3_3.copyparams(pretrained_model.conv3_3)
        self.extractor.conv4_1.copyparams(pretrained_model.conv4_1)
        self.extractor.conv4_2.copyparams(pretrained_model.conv4_2)
        self.extractor.conv4_3.copyparams(pretrained_model.conv4_3)
        self.extractor.conv5_1.copyparams(pretrained_model.conv5_1)
        self.extractor.conv5_2.copyparams(pretrained_model.conv5_2)
        self.extractor.conv5_3.copyparams(pretrained_model.conv5_3)
        self.head.fc6.copyparams(pretrained_model.fc6)
        self.head.fc7.copyparams(pretrained_model.fc7)
        self.head.lstm8.copyparams(pretrained_model.fc8)



class VGG16RoIHead(chainer.Chain):

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

    def __init__(self,  roi_size, spatial_scale,
                 vgg_initialW=None,use_roi_align=False):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        self.use_roi_align = use_roi_align
        with self.init_scope():
            self.fc6 = L.Linear(25088, 4096, initialW=vgg_initialW)
            self.fc7 = L.Linear(4096, 2048, initialW=vgg_initialW)
        self.functions = collections.OrderedDict([
            ('fc6', [self.fc6, F.relu]),
            ("fc7", [self.fc7]),
        ])
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.activation = dict()



    def __call__(self, x, rois, roi_indices, layers=["fc"]):
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
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)  # None means np.newaxis, concat along column
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale, self.use_roi_align)
        h = pool
        for key, funcs in self.functions.items():
            for func in funcs:
                h = func(h)
        return h


class VGG16FeatureExtractor(chainer.Chain):
    """Truncated VGG-16 that extracts a conv5_3 feature map.

    Args:
        initialW (callable): Initializer for the weights.

    """

    def __init__(self, initialW=None):
        super(VGG16FeatureExtractor, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1, initialW=initialW)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, initialW=initialW)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1, initialW=initialW)
            self.conv2_2 = L.Convolution2D(
                128, 128, 3, 1, 1, initialW=initialW)
            self.conv3_1 = L.Convolution2D(
                128, 256, 3, 1, 1, initialW=initialW)
            self.conv3_2 = L.Convolution2D(
                256, 256, 3, 1, 1, initialW=initialW)
            self.conv3_3 = L.Convolution2D(
                256, 256, 3, 1, 1, initialW=initialW)
            self.conv4_1 = L.Convolution2D(
                256, 512, 3, 1, 1, initialW=initialW)
            self.conv4_2 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)
            self.conv4_3 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)
            self.conv5_1 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)
            self.conv5_2 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)
            self.conv5_3 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)

        self.functions = collections.OrderedDict([
            ('conv1_1', [self.conv1_1, F.relu]),
            ('conv1_2', [self.conv1_2, F.relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2_1', [self.conv2_1, F.relu]),
            ('conv2_2', [self.conv2_2, F.relu]),
            ('pool2', [_max_pooling_2d]),
            ('conv3_1', [self.conv3_1, F.relu]),
            ('conv3_2', [self.conv3_2, F.relu]),
            ('conv3_3', [self.conv3_3, F.relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [self.conv4_1, F.relu]),
            ('conv4_2', [self.conv4_2, F.relu]),
            ('conv4_3', [self.conv4_3, F.relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [self.conv5_1, F.relu]),
            ('conv5_2', [self.conv5_2, F.relu]),
            ('conv5_3', [self.conv5_3, F.relu]),
        ])
        self.activation = {}

    def __call__(self, x):
        h = x
        for idx, (key, funcs) in enumerate(self.functions.items()):
            for func in funcs:
                h = func(h)
        return h


def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale, use_roi_align):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    if use_roi_align:
        pool = roi_align_2d(x, xy_indices_and_rois, outh, outw, spatial_scale)
    else:
        pool = F.roi_pooling_2d(x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)
