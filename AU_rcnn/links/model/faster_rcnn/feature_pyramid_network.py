import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNet101Layers
import config
import numpy as np
from lstm_end_to_end.model.AU_rcnn.roi_tools.roi_align_2d import roi_align_2d
from chainer import Sequential
from AU_rcnn.transforms.image.resize import resize

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


class FPN101(chainer.Chain):
    _models = {
        'resnet101': {
            'path': config.ROOT_PATH + '/caffe_model/ResNet-101-model.npz'
        }
    }
    def __init__(self, classes, pretrained_resnet, use_roialign=False, mean_path="", min_size=512, max_size=512):
        super(FPN101, self).__init__()
        self.classes = classes
        self.use_roialign = use_roialign

        self.min_size = min_size
        self.max_size = max_size

        mean = np.load(mean_path)
        self.mean = np.resize(mean, (3, self.min_size, self.min_size))


        with self.init_scope():
            self.top_layer = L.Convolution2D(2048, 256, ksize=1, stride=1, pad=0) # Reduce channels
            # smooth layer
            self.smooth1 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1)
            self.smooth2 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1)
            self.smooth3 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1)
            # Lateral layers
            self.latlayer1 = L.Convolution2D(1024, 256, ksize=1, stride=1, pad=0)
            self.latlayer2 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0)
            self.latlayer3 = L.Convolution2D(256, 256, ksize=1, stride=1, pad=0)
            self.roi_feat_downsample = L.Convolution2D(256, 256, ksize=3, stride=2, pad=1)

            self.rcnn_top = Sequential(L.Convolution2D(256, 1024, ksize=7, stride=7, pad=0),
                                       F.relu, L.Convolution2D(1024, 1024, ksize=1, stride=1, pad=0), F.relu)
            self.cls_score = L.Linear(1024, self.classes)
            self.resnet101 = ResNet101Layers(pretrained_model=self._models["resnet101"]["path"])
            if pretrained_resnet.endswith(".npz"):
                print("loading :{} to FPN-101".format(pretrained_resnet))
                chainer.serializers.load_npz(pretrained_resnet, self)

    def _head_to_tail(self, pool5):
        block5 = self.rcnn_top(pool5) # B, 1024, 1, 1
        fc7 = F.mean(F.mean(block5, 3),2) # B, 1024
        return fc7

    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.resize_images(x, output_shape=(H, W)) + y

    def _pyramid_roi_feature(self, feature_maps, rois):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        h = rois[:, 3] - rois[:, 1] + 1  # batch_index, y_min, x_min, y_max, x_max
        w = rois[:, 4] - rois[:, 2] + 1
        h = chainer.cuda.to_cpu(h)
        w = chainer.cuda.to_cpu(w)
        roi_level = np.log(np.sqrt(h * w)/ 224.0)
        roi_level = np.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5

        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(2, 6)):
            if np.sum(roi_level == l) == 0:
                continue
            idx_l = np.nonzero(roi_level == l)[0] # 该层的box的index
            box_to_levels.append(idx_l)
            scale = feature_maps[i].shape[2] / config.IMG_SIZE[0]
            feat = _roi_pooling_2d_yx(feature_maps[i], rois[idx_l], outh=7, outw=7,spatial_scale=scale,
                                          use_roi_align=self.use_roialign)   # 第i个层的featuremap选用第idx_l个box
            roi_pool_feats.append(feat)  # 可见原本是按照金字塔的 [p2, p3, p4, p5]来排序的顺序
        roi_pool_feats = F.concat(roi_pool_feats, 0)
        box_to_levels = np.concatenate(box_to_levels, 0)  # 各层的box index
        order = np.argsort(np.asarray(box_to_levels)) #按照box的index排序得到order
        roi_pool_feats = roi_pool_feats[order]  # 按box的index重新排序
        return roi_pool_feats, box_to_levels

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

    def predict(self, imgs, bboxes):
        assert bboxes.shape[-1] == 4
        xp = chainer.cuda.get_array_module(imgs)
        _, _, H, W = imgs.shape
        rois = list()
        roi_indices = list()
        for n in range(imgs.shape[0]):  # n is batch index
            bbox = bboxes[n].data

            bbox[:, 0::2] = xp.clip(
                bbox[:, 0::2], 0, H)  # 抄来的一句话
            bbox[:, 1::2] = xp.clip(
                bbox[:, 1::2], 0, W)

            rois.extend(bbox.tolist())
            roi_indices.extend((n * xp.ones(bbox.shape[0])).tolist())
        rois = xp.asarray(rois, dtype=xp.float32)  # shape = R,4
        roi_indices = xp.asarray(roi_indices, dtype=xp.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)

        roi_scores = self.__call__(imgs, indices_and_rois)
        pred_label = self.fetch_labels_from_scores(xp, roi_scores.data)
        return pred_label, roi_scores.data


    def __call__(self, x, boxes):
        assert boxes.shape[1] == 5
        # boxes shape
        mini_batch = x.shape[0]
        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.resnet101.conv1(x)
        c1 = self.resnet101.bn1(c1)
        c2 = self.resnet101.res2(c1)
        c3 = self.resnet101.res3(c2)
        c4 = self.resnet101.res4(c3)  # channel = 1024
        c5 = self.resnet101.res5(c4) # channel = 2048

        # Top-down
        p5 = self.top_layer(c5) # channel = 256
        p4 = self._upsample_add(p5, self.latlayer1(c4))  # channel = 256
        p4 = self.smooth1(p4)  # 3 x 3 conv -> channel = 256
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        mrcnn_feature_maps = [p2, p3, p4, p5]
        roi_pool_feats, box_to_levels = self._pyramid_roi_feature(mrcnn_feature_maps, boxes)
        pooled_feat = self._head_to_tail(roi_pool_feats)
        cls_score = self.cls_score(pooled_feat)  # R', 12
        # cls_score = cls_score.reshape(mini_batch, -1, self.classes)

        return cls_score