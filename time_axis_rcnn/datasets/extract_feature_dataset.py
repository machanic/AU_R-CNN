from collections import defaultdict

from AU_rcnn.datasets.AU_dataset import AUDataset
import chainer
import numpy as np

class FeatureExtractorDataset(AUDataset):

    def __init__(self, database, fold, split_name, split_index, mc_manager, use_lstm, train_all_data, prefix="",
                 pretrained_target="", pretrained_model=None, extract_key="avg_pool", device=-1, batch_size=5):
        super(FeatureExtractorDataset, self).__init__(database, fold, split_name, split_index, mc_manager, use_lstm,
                                                      train_all_data, prefix, pretrained_target)
        self.extract_key = extract_key
        self.model = pretrained_model
        self.device = device
        if device is None:
            def to_device(x):
                return x
        elif device < 0:
            to_device = chainer.cuda.to_cpu
        else:
            def to_device(x):
                return chainer.cuda.to_gpu(x, device, chainer.cuda.Stream.null)
        self.to_device = to_device
        self.batch_pieces = []
        self.seq_dict = defaultdict(list)
        for idx, (img_path, from_img_path, AU_set, current_database_name) in enumerate(self.result_data):
            sequence_key = "_".join((img_path.split("/")[-3], img_path.split("/")[-2]))
            self.seq_dict[sequence_key].append(idx)
        for sequence_key, entry_list in self.seq_dict.items():
            sublist = [entry_list[i:i + batch_size] for i in range(0, len(entry_list), batch_size)]
            self.batch_pieces.extend(sublist)

    def __len__(self):
        return len(self.batch_pieces)

    def get_example(self, i):
        if i > len(self):
            raise IndexError("Index too large , i = {}".format(i))
        batch_list = self.batch_pieces[i]
        sequence_key = None
        all_faces = []
        boxes = []
        labels = []
        for idx in batch_list:
            img_path, from_img_path, AU_set, current_database_name = self.result_data[idx]
            sequence_key = "_".join((img_path.split("/")[-3], img_path.split("/")[-2]))
            try:
                cropped_face, bbox, label, AU_couple_lst = super(FeatureExtractorDataset, self).get_example(idx)
                all_faces.append(cropped_face)
                boxes.append(bbox)
                labels.append(label)
            except Exception:
                print("error crop in image path : {}".format(img_path))
                continue

        if not all_faces:
            return None, None, sequence_key
        all_faces = np.stack(all_faces)  # B, C, H, W
        boxes = np.stack(boxes) # B, F, 4
        labels = np.stack(labels)  # B, F, 4

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # TODO, 光流没有实现extract方法
            feature = self.model.extract_batch(all_faces, boxes, layer=self.extract_key) # shape = R', 2048

        return self.to_device(feature), labels, sequence_key