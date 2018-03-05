from AU_rcnn.datasets.AU_dataset import AUDataset
from collections import defaultdict
import chainer
import numpy as np

class AUExtractorDataset(AUDataset):


    def __init__(self, database, fold, split_name, split_index, mc_manager, use_lstm, train_all_data, prefix="",
                 pretrained_target="", pretrained_model=None, extract_key="avg_pool", device=-1, batch_size=200):
        super(AUExtractorDataset, self).__init__(database, fold, split_name, split_index, mc_manager, use_lstm, train_all_data,
                                        prefix, pretrained_target)
        self.model = pretrained_model
        self.device = device
        self.extract_key = extract_key
        if device is None:
            def to_device(x):
                return x
        elif device < 0:
            to_device = chainer.cuda.to_cpu
        else:
            def to_device(x):
                return chainer.cuda.to_gpu(x, device, chainer.cuda.Stream.null)
        self.to_device = to_device

        self.batch_size = batch_size

        self.sequence_name_list = []
        for img_path, *_ in self.result_data:
            sequence_id = "/".join((img_path.split("/")[-3], img_path.split("/")[-2]))
            self.sequence_name_list.append(sequence_id)



    def get_example(self, i):
        if i > len(self):
            raise IndexError("Index too large , i = {}".format(i))
        img_path, from_img_path, AU_set, database_name = self.result_data[i]
        sequence_key = "_".join((img_path.split("/")[-3], img_path.split("/")[-2]))
        try:
            cropped_face, bbox, label, AU_couple_lst = super(AUExtractorDataset, self).get_example(i)
        except Exception:
            return None, None, None, img_path

        if cropped_face is None or cropped_face.ndim != 3:
            print("not found image, because landmark error in {}".format(img_path))
            return None, None, None, img_path
        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            feature = self.model.extract(cropped_face, bbox, layer=self.extract_key) # shape = R', 2048
            feature = feature.reshape(bbox.shape[0], -1)
        cropped_face = None

        return self.to_device(feature), bbox, label, img_path