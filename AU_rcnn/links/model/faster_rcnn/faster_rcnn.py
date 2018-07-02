# Mofidied work:
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# --------------------------------------------------------
#
# Original works by:
# --------------------------------------------------------
# Faster R-CNN implementation by Chainer
# Copyright (c) 2016 Shunta Saito
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/mitmul/chainer-faster-rcnn
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

from __future__ import division

import chainer
import cv2
import numpy as np
from chainer import cuda

from AU_rcnn import transforms
from AU_rcnn.links.model.faster_rcnn.utils.proposal_target_creator import ProposalTargetCreator
from AU_rcnn.transforms.image.resize import resize

class FasterRCNN(chainer.Chain):

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
        super(FasterRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.head = head

        self.mean = np.resize(mean,(3,min_size,min_size))
        self.min_size = min_size
        self.max_size = max_size
        self.proposal_target_creator = ProposalTargetCreator()
        self.extract_dict = dict()

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def extract(self, image, bbox, layer='res5'): # image shape is C,H,W, bboxes is R,4 where R is box number inside one image
        _, H, W = image.shape
        x = self.prepare(image)
        _, o_H, o_W = x.shape
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        assert len(np.where(bbox < 0)[0]) == 0
        x = chainer.Variable(self.xp.asarray([x]))  # shape = (1,C,H,W)  # look https://github.com/chainer/chainer/issues/1087
        bbox = chainer.Variable(self.xp.asarray([bbox])) # shape = (1, box_num, 4)
        roi_scores, rois, roi_indices = self.__call__(x, bbox, layers=[layer])
        feature = self.extract_dict[layer]  # shape = R' x 2048 where R' is number bbox
        self.extract_dict.clear()
        return feature

    def extract_batch(self, images, bboxes, layer='res5'):  # images = batch x C x H x W
        xs = []
        for image in images:
            x = self.prepare(image)
            xs.append(x)
        xs = np.stack(xs)
        xs = chainer.Variable(self.xp.asarray(xs))         # shape = (B, C, H, W)
        bboxes = chainer.Variable(self.xp.asarray(bboxes))  # shape = (B, F, 4)
        mini_batch, frame_box_num, _ = bboxes.shape
        roi_scores, rois, roi_indices = self.__call__(xs, bboxes, layers=[layer])
        feature = self.extract_dict[layer]  # shape = R' x 2048 where R' is number bbox across batch
        self.extract_dict.clear()
        feature = feature.reshape(mini_batch, frame_box_num, feature.shape[1])
        return feature


    # 预测的时候才可能被调用，train的时候反而不调用，具体可看faster_rcnn_train_chain.py
    # 若两个box的IOU较大，将较大面积的box删掉
    def __call__(self, x, bboxes, layers=["avg_pool"]): # 预测的时候同样要提供来自于landmark的bounding box
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
        assert x.shape[0] == bboxes.shape[0]
        _, _, H, W = x.shape
        h = self.extractor(x)
        rois = list()
        roi_indices = list()
        for n in range(x.shape[0]): # n is batch index
            bbox = bboxes[n].data

            bbox[:, 0::2] = xp.clip(
                bbox[:, 0::2], 0, H)  # 抄来的一句话
            bbox[:,  1::2] = xp.clip(
                bbox[:,  1::2], 0, W)

            rois.extend(bbox.tolist())
            roi_indices.extend((n * xp.ones(bbox.shape[0])).tolist())
        rois = xp.asarray(rois, dtype=xp.float32)  # shape = R,4
        roi_indices = xp.asarray(roi_indices, dtype=xp.int32)
        roi_scores = self.head(h, rois, roi_indices, layers=layers)  # roi_scores is each roi has vector of scores!
        self.extract_dict.update({layer: self.head.activation[layer].data for layer in layers})
        self.head.activation.clear()
        return roi_scores, rois, roi_indices


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

    def fetch_labels_from_scores(self, xp, raw_score):
        '''
        R' is bbox count in one image
        :param raw_score: shape = (R', L)
        :return:
        '''
        pred_labels = xp.where(raw_score > 0, 1, 0).astype(xp.int32)
        return pred_labels  # note that pred_labels and scores are list of list, not np.ndarray.


    def predict(self, imgs, bbox): # 传入的是一个batch的数据
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
            roi_scores, _, _ = self.__call__(imgs, bbox, layers=["fc"]) # shape = R', class_num
            pred_label = self.fetch_labels_from_scores(xp, roi_scores.data)
        return pred_label, roi_scores.data

