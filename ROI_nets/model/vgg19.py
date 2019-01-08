import collections
from collections import defaultdict
from functools import partial

import chainer
import chainer.functions as F
import chainer.links as L
import cv2
import numpy as np
from chainer import cuda

import config
from AU_rcnn.transforms.image.resize import resize

class ROINetBase(chainer.Chain):

    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.

    Each stage is carried out by one of the callable
    :class:`chainer.Chain` objects :obj:`feature`, :obj:`rpn` and :obj:`head`.

    There are two func :meth:`predict` and :meth:`__call__` to conduct
    object detection.
    :meth:`predict` takes images and returns bounding boxes that are converted
    to image coordinates. This will be useful for a scenario when
    Faster R-CNN is treated as a black box legacy, for instance.
    :meth:`__call__` is provided for a scnerario when intermediate outputs
    are needed, for instance, for training and debugging.

    Links that support obejct detection API have method :meth:`predict` with
    the same interface. Please refer to :func:`FasterRCNN.predict` for
    further details.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        extractor (callable Chain): A callable that takes a BCHW image
            array and returns feature maps.
        head (callable Chain): A callable that takes
            a BCHW array, RoIs and batch indices for RoIs. This returns class
            class scores.
        mean (numpy.ndarray): A value to be subtracted from an image
            in :meth:`prepare`.
        min_size (int): A preprocessing paramter for :meth:`prepare`. Please
            refer to a docstring found for :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.

    """

    def __init__(
            self, extractor, head,
            mean,
            min_size=512,
            max_size=512,
    ):
        super(ROINetBase, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.head = head

        self.mean = np.resize(mean,(3,min_size,min_size))
        self.min_size = min_size
        self.max_size = max_size
        self.extract_dict = dict()

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class



    # 预测的时候才可能被调用，train的时候反而不调用，具体可看faster_rcnn_train_chain.py
    # 若两个box的IOU较大，将较大面积的box删掉
    def __call__(self, x): # 预测的时候同样要提供来自于landmark的bounding box
        """Forward Faster R-CNN.
        called by self.predict

        Here are notations used.

        * :math:`N` is the number of batch size
        * :math:`R'` is the total number of RoIs produced across batches. \
            Given :math:`R_i` proposed RoIs from the :math:`i` th image, \
            :math:`R' = \\sum _{i=1} ^ N R_i`.
        * :math:`L` is the number of classes.

        Classes are ordered by the background, the first class, ..., and
        the :math:`L` th class.

        Args:
            x (~chainer.Variable): 4D image variable (N, C, H, W).
            bboxes (~chainer.Variable): variable (N, R, 4). shape[1] of bbox is bboxes which is in one image

        Returns:
            Variable, Variable, array, array:
            Returns tuple of four values listed below.

            * **roi_scores**: Class predictions for the proposed RoIs. \
                Its shape is :math:`(R', num_classes)`.
            * **rois**: RoIs proposed by RPN. Its shape is \
                :math:`(R', 4)`.
            * **roi_indices**: Batch indices of RoIs. Its shape is \
                :math:`(R',)`.

        """
        xp = cuda.get_array_module(x)
        _, _, H, W = x.shape
        h = self.extractor(x)
        roi_scores = self.head(h)  # roi_scores is each roi has vector of scores!

        return roi_scores


    def prepare(self, img):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`self.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        _, H, W = img.shape

        scale = self.min_size / min(H, W)

        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)

        img = resize(img, (int(H * scale), int(W * scale)))

        img = (img - self.mean).astype(np.float32, copy=False)
        return img

    def predict(self, imgs): # 传入的是一个batch的数据
        """predict AUs from image and bbox.

        Args:
            imgs(iterable of numpy.ndarray): Arrays holding images. shape = (B,C,H,W), where B is batch_size
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.
            bboxes (iterable of numpy.ndarray): Arrays holding bounding boxes, each is bboxes inside one image shape=(Batch,R,4)

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.
        """
        xp = cuda.get_array_module(imgs)

        with cuda.get_device_from_array(imgs) as device, \
                chainer.function.no_backprop_mode():
            roi_scores = self.__call__(imgs) # shape = R', class_num
        return roi_scores.data


class ROI_NetsVGG19(ROINetBase):

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
    :class:`AU_rcnn.links.model.faster_rcnn.FasterRCNN`.

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
            the VGG-19 layers.
        rpn_initialW (callable): Initializer for Region Proposal Network
            layers.
        loc_initialW (callable): Initializer for the localization head.
        score_initialW (callable): Initializer for the score head.
        proposal_creator_params (dict): Key valued paramters for
            :obj:`AU_rcnn.links.model.faster_rcnn.ProposalCreator`.

    """


    def __init__(self,
                 pretrained_model=None,
                 min_size=config.IMG_SIZE[0], max_size=config.IMG_SIZE[0],
                 vgg_initialW=None,
                 mean_file=None):

        n_fg_class = len(config.AU_SQUEEZE)
        if vgg_initialW is None and pretrained_model:
            vgg_initialW = chainer.initializers.LeCunUniform(scale=1.0)

        extractor = VGG19FeatureExtractor(initialW=vgg_initialW)
        head = VGG19RoIHead(
            n_fg_class,  # 注意:全0表示背景。010101才表示多label，因此无需一个特别的0的神经元节点
            roi_size=3,  # 1/ 16.0 means after extract feature map, the map become 1/16 of original image, ROI bbox also needs shrin
            roi_count=20
        )
        self.mean_file = mean_file
        mean_array = np.load(mean_file)
        print("loading mean_file in: {} done".format(mean_file))
        super(ROI_NetsVGG19, self).__init__(
            extractor,
            head,
            mean=mean_array,
            min_size=min_size,
            max_size=max_size
        )


class VGG19RoIHead(chainer.Chain):

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

    def __init__(self, n_class, roi_size, roi_count=20):
        # n_class includes the background
        super(VGG19RoIHead, self).__init__()
        self.functions = defaultdict(list)
        self.roi_count = roi_count
        with self.init_scope():
            for roi_idx in range(roi_count):
                self.functions[roi_idx].append(partial(F.resize_images, output_shape=(6,6)))
                setattr(self, "{}_resize".format(roi_idx), self.functions[roi_idx][-1])
                self.functions[roi_idx].append(L.Convolution2D(512, 512, 3, 1, 1, False))
                setattr(self, "{}_roi_conv1".format(roi_idx), self.functions[roi_idx][-1])
                self.functions[roi_idx].append(F.relu)
                setattr(self, "{}_roi_relu1".format(roi_idx), self.functions[roi_idx][-1])
                self.functions[roi_idx].append(L.Convolution2D(512, 512, 3, 1, 1, False))
                setattr(self, "{}_roi_conv2".format(roi_idx), self.functions[roi_idx][-1])
                self.functions[roi_idx].append(F.relu)
                setattr(self, "{}_roi_relu2".format(roi_idx), self.functions[roi_idx][-1])
                self.functions[roi_idx].append(partial(F.reshape, shape=(-1, 18432)))
                setattr(self, "{}_roi_reshape".format(roi_idx), self.functions[roi_idx][-1])
                self.functions[roi_idx].append(L.Linear(18432, 150))
                setattr(self, "{}_roi_fc".format(roi_idx), self.functions[roi_idx][-1])

            self.fc1 = L.Linear(3000,2048)
            self.fc2 = L.Linear(2048, n_class)
        self.n_class = n_class
        self.roi_size = roi_size



    def __call__(self, x):
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
        height, width = x.shape[2], x.shape[3]
        rois = np.random.randint(0, height-5, size=(self.roi_count, 2), dtype=np.int32) #FIXME
        new_rois = rois + self.roi_size
        rois = np.concatenate([rois, new_rois],axis=1)
        all_roi_feature = []
        for idx in range(self.roi_count):
            roi = rois[idx]  # y_min, x_min, y_max, x_max
            roi_fm = x[:, :, roi[0] : roi[2], roi[1]: roi[3]]  # B, C, H, W
            assert roi_fm.shape[-1] ==self.roi_size
            funcs = self.functions[idx]
            for func in funcs:
                roi_fm = func(roi_fm) # batch, 150
            all_roi_feature.append(roi_fm)
        all_roi_feature = F.concat(all_roi_feature, axis=1)
        all_roi_feature = self.fc1(all_roi_feature)
        all_roi_feature = self.fc2(all_roi_feature)

        return all_roi_feature


class VGG19FeatureExtractor(chainer.Chain):
    """Truncated VGG-16 that extracts a conv5_3 feature map.

    Args:
        initialW (callable): Initializer for the weights.

    """

    def __init__(self, initialW=None):
        super(VGG19FeatureExtractor, self).__init__()
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
            self.conv3_4 = L.Convolution2D(256, 256, 3, 1, 1, initialW=initialW)

            self.conv4_1 = L.Convolution2D(
                256, 512, 3, 1, 1, initialW=initialW)
            self.conv4_2 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)
            self.conv4_3 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)
            self.conv4_4 = L.Convolution2D(512, 512, 3, 1, 1,initialW=initialW)
            self.conv5_1 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)
            self.conv5_2 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)
            self.conv5_3 = L.Convolution2D(
                512, 512, 3, 1, 1, initialW=initialW)
            self.conv5_4 = L.Convolution2D(512, 512, 3, 1, 1, initialW=initialW)

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
            ('conv3_4', [self.conv3_4, F.relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [self.conv4_1, F.relu]),
            ('conv4_2', [self.conv4_2, F.relu]),
            ('conv4_3', [self.conv4_3, F.relu]),
            ('conv4_4', [self.conv4_4, F.relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [self.conv5_1, F.relu]),
            ('conv5_2', [self.conv5_2, F.relu]),
            ('conv5_3', [self.conv5_3, F.relu]),
            ('conv5_4', [self.conv5_4, F.relu])
        ])
        self.activation = {}

    def __call__(self, x,  layers=[None]):
        target_layers = set(layers)
        h = x

        for idx, (key, funcs) in enumerate(self.functions.items()):
            for func in funcs:
                h = func(h)
        return h


def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = F.roi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=2)
