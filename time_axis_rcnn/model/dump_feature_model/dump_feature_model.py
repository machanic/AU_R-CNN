import copy
import os
from collections import defaultdict

import chainer
import numpy as np
from chainer.training.extensions import Evaluator
from overrides import overrides

import config


class DumpRoIFeature(Evaluator):

    trigger = 1, 'epoch'
    default_name = 'DumpRoIFeature'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, model, device, database, converter,
                 output_path, trainval_test, fold_split_idx, mirror_data=False):

        super(DumpRoIFeature, self).__init__(iterator, model, device=device, converter=converter)
        self.database = database
        self.paper_use_AU = []
        self.output_path = output_path
        self.trainval_test = trainval_test
        self.fold_split_idx = fold_split_idx
        self.mirror_data = mirror_data

    def get_npz_name(self, AU_group_id, trainval_test, out_dir, database, fold, split_idx, sequence_key):

        if trainval_test == "trainval":
            train_keyword = "train"
            if self.mirror_data:
                train_keyword = "train_mirror"
            file_name = out_dir + os.path.sep + "{0}_{1}_fold_{2}".format(database, fold,
                                                                          split_idx) \
                        + "/{}/".format(train_keyword) + sequence_key + "#{}.npz".format(
                AU_group_id)
        else:
            file_name = out_dir + os.path.sep + "{0}_{1}_fold_{2}".format(database, fold,
                                                                          split_idx) \
                        + "/test/" + sequence_key + "#{}.npz".format(
                AU_group_id)
        return file_name

    def dump_feature(self, rgb_feature_list, flow_feature_list,
                     fuse_feature_list, labels_list, seq_key):

        fuse_trans = defaultdict(list)
        rgb_trans = defaultdict(list)
        flow_trans = defaultdict(list)

        for fuse_feature, rgb_feature, flow_feature in zip(fuse_feature_list, rgb_feature_list, flow_feature_list):
            for AU_group_id, (rgb, flow, fuse) in enumerate(zip(rgb_feature, flow_feature, fuse_feature)):
                fuse_trans[AU_group_id].append(fuse)
                rgb_trans[AU_group_id].append(rgb)
                flow_trans[AU_group_id].append(flow)

        label = np.stack(labels_list)  # shape = B, F, 12
        # fuse_trans = np.transpose(fuse_feature, axes=(1, 0, 2))  # shape = F, B, 2048
        # rgb_trans = np.transpose(rgb_feature, axes=(1,0,2))  # shape = F, B, 2048
        # flow_trans = np.transpose(flow_feature, axes=(1,0,2))  # shape = F, B, 2048

        labels_trans = np.transpose(label, axes=(1, 0, 2))  # shape = F, N, 12
        for AU_group_id, fuse_box_feature_list in fuse_trans.items():
            fuse_box_feature = np.stack(fuse_box_feature_list)
            flow_box_feature = np.stack(flow_trans[AU_group_id])
            rgb_box_feature = np.stack(rgb_trans[AU_group_id])

            count_non_zero = len(np.nonzero(labels_trans[AU_group_id])[0])
            if count_non_zero == 0:
                continue
            out_filepath = self.get_npz_name(AU_group_id, self.trainval_test, self.output_path, self.database,
                                             3,  self.fold_split_idx, seq_key)

            print("write : {0}".format(out_filepath))

            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            np.savez(out_filepath, fuse_feature=fuse_box_feature, rgb_feature=rgb_box_feature,
                     flow_feature=flow_box_feature,
                     label=labels_trans[AU_group_id])
        fuse_feature_list.clear()
        rgb_feature_list.clear()
        flow_feature_list.clear()
        labels_list.clear()

    def extract_sequence_key(self, img_path):
        return "_".join((img_path.split("/")[-3], img_path.split("/")[-2]))

    @overrides
    def evaluate(self):
        iterator = self._iterators['main']
        _target = self._targets["main"]
        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        model = _target
        last_sequence_key = None

        rgb_roi_feature_list = []
        flow_roi_feature_list = []
        fuse_roi_feature_list = []

        roi_labels_list = []

        for idx, batch in enumerate(it):
            delete_rgb_row = []
            print("processing :{}".format(idx))
            batch = self.converter(batch, -1)
            rgb_faces, flow_faces, bboxes, labels, rgb_path_list = batch  # rgb_faces shape = B, C, H, W; bboxes shape = B, F, 4; labels shape = B, F, 12
            xp = chainer.cuda.get_array_module(rgb_faces)
            last_rgb_path = rgb_path_list[0]
            for idx, rgb_path in enumerate(rgb_path_list[1:]):
                if last_rgb_path == rgb_path:
                    delete_rgb_row.append(idx + 1)
                last_rgb_path = rgb_path

            if delete_rgb_row:
                rgb_faces = np.delete(rgb_faces, delete_rgb_row, axis=0)
                bboxes = np.delete(bboxes, delete_rgb_row, axis=0)
                labels = np.delete(labels, delete_rgb_row, axis=0)


            sequence_key = self.extract_sequence_key(rgb_path_list[0])

            if last_sequence_key is None:
                last_sequence_key = sequence_key

            if not isinstance(rgb_faces, chainer.Variable):
                rgb_faces = chainer.Variable(chainer.cuda.to_gpu(rgb_faces.astype('f'), self.device))
                flow_faces = chainer.Variable(chainer.cuda.to_gpu(flow_faces.astype('f'), self.device))
                bboxes = chainer.Variable(chainer.cuda.to_gpu(bboxes.astype('f'), self.device))
            if sequence_key != last_sequence_key:  # 换video了
                if len(fuse_roi_feature_list) == 0:
                    print("all feature cannot obtain {}".format(last_sequence_key))
                self.dump_feature(rgb_roi_feature_list,flow_roi_feature_list,
                                  fuse_roi_feature_list, roi_labels_list, last_sequence_key)
                last_sequence_key = sequence_key

            # roi_feature_rgb = (B*T, F, 2048)    roi_feature_flow = (B, F, 2048)
            if labels is not None and labels.shape[1] == config.BOX_NUM[self.database]:
                rgb_feature, flow_feature, roi_feature = model.extract_batch(rgb_faces, flow_faces, bboxes) # # B, F, 2048
                rgb_feature = chainer.cuda.to_cpu(rgb_feature.data)
                flow_feature = chainer.cuda.to_cpu(flow_feature.data)
                roi_feature = chainer.cuda.to_cpu(roi_feature.data)
                rgb_roi_feature_list.extend(rgb_feature)
                flow_roi_feature_list.extend(flow_feature)
                fuse_roi_feature_list.extend(roi_feature)
                roi_labels_list.extend(labels)

        # the last sequence key
        self.dump_feature(rgb_roi_feature_list,flow_roi_feature_list,
                          fuse_roi_feature_list, roi_labels_list, last_sequence_key)