import copy
import os

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
                 output_path, trainval_test, fold_split_idx):
        super(DumpRoIFeature, self).__init__(iterator, model, device=device, converter=converter)
        self.database = database
        self.paper_use_AU = []
        self.output_path = output_path
        self.trainval_test = trainval_test
        self.fold_split_idx = fold_split_idx


    def get_npz_name(self, AU_group_id, trainval_test, out_dir, database, fold, split_idx, sequence_key):
        if trainval_test == "trainval":
            file_name = out_dir + os.path.sep + "{0}_{1}_fold_{2}".format(database, fold,
                                                                          split_idx) \
                        + "/train/" + sequence_key + "#{}.npz".format(
                AU_group_id)
        else:
            file_name = out_dir + os.path.sep + "{0}_{1}_fold_{2}".format(database, fold,
                                                                          split_idx) \
                        + "/test/" + sequence_key + "#{}.npz".format(
                AU_group_id)
        return file_name

    def dump_feature(self, roi_feature_rgb_list, roi_feature_flow_list, labels_list, seq_key):

        feature_rgb = np.stack(roi_feature_rgb_list)  # shape = B*T, F, 2048
        feature_flow = np.stack(roi_feature_flow_list)  # shape = B, F, 2048
        label = np.stack(labels_list)  # shape = B*T, F, 12
        feature_rgb_trans = np.transpose(feature_rgb, axes=(1, 0, 2))  # shape = F, B*T, 2048
        feature_flow_trans = np.transpose(feature_flow, axes=(1, 0, 2))  # shape = F, B, 2048
        labels_trans = np.transpose(label, axes=(1, 0, 2))  # shape = F, N, 12
        assert feature_flow_trans.shape[0] == feature_rgb_trans.shape[0] == labels_trans.shape[0]
        for AU_group_id, rgb_box_feature in enumerate(feature_rgb_trans):
            out_filepath = self.get_npz_name(AU_group_id, self.trainval_test, self.output_path, self.database,
                                             3,
                                             self.fold_split_idx, seq_key)
            flow_box_feature = feature_flow_trans[AU_group_id]  # (B, 2048), and feature_rgb_trans = (B*T, 2048)
            print("write : {0}".format(out_filepath))

            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

            np.savez(out_filepath, rgb_feature=rgb_box_feature, flow_feature=flow_box_feature,
                     label=labels_trans[AU_group_id])
        roi_feature_rgb_list.clear()
        roi_feature_flow_list.clear()
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

        roi_feature_rgb_list = []
        roi_feature_flow_list = []
        roi_labels_list = []


        for idx, batch in enumerate(it):
            delete_rgb_row = []
            print("processing :{}".format(idx))
            batch = self.converter(batch, -1)
            rgb_faces, flow_faces, bboxes, labels_list, rgb_path_list = batch  # images shape = B*T, C, H, W; bboxes shape = B*T, F, 4; labels shape = B*T, F, 12
            xp = chainer.cuda.get_array_module(rgb_faces)
            last_rgb_path = rgb_path_list[0]
            for idx, rgb_path in enumerate(rgb_path_list[1:]):
                if last_rgb_path == rgb_path:
                    delete_rgb_row.append(idx + 1)
                last_rgb_path = rgb_path

            if delete_rgb_row:
                rgb_faces = np.delete(rgb_faces, delete_rgb_row, axis=0)
                bboxes = np.delete(bboxes, delete_rgb_row, axis=0)
                new_label_list = []
                for idx, label in enumerate(labels_list):
                    if idx not in delete_rgb_row:
                        new_label_list.append(label)
                labels = np.stack(new_label_list)
            else:
                labels = np.stack(labels_list)

            sequence_key = self.extract_sequence_key(rgb_path_list[0])

            if last_sequence_key is None:
                last_sequence_key = sequence_key

            if not isinstance(rgb_faces, chainer.Variable):
                rgb_faces = chainer.Variable(chainer.cuda.to_gpu(rgb_faces.astype('f'), self.device))
                flow_faces = chainer.Variable(chainer.cuda.to_gpu(flow_faces.astype('f'), self.device))
                bboxes = chainer.Variable(chainer.cuda.to_gpu(bboxes.astype('f'), self.device))

            if sequence_key != last_sequence_key:  # 换video了
                if len(roi_feature_rgb_list) == 0:
                    print("all feature cannot obtain {}".format(last_sequence_key))
                self.dump_feature(roi_feature_rgb_list, roi_feature_flow_list, roi_labels_list, last_sequence_key)
                last_sequence_key = sequence_key

            # roi_feature_rgb = (B*T, F, 2048)    roi_feature_flow = (B, F, 2048)
            if labels is not None and labels.shape[1] == config.BOX_NUM[self.database]:
                roi_feature_rgb, roi_feature_flow = model.extract_batch(rgb_faces, flow_faces, bboxes, None)
                roi_feature_rgb_list.extend(roi_feature_rgb)
                roi_feature_flow_list.extend(roi_feature_flow)
                roi_labels_list.extend(labels)

        # the last sequence key
        self.dump_feature(roi_feature_rgb_list, roi_feature_flow_list, roi_labels_list, last_sequence_key)