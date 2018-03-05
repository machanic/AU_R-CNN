import chainer
import os
from collections import defaultdict


class JumpExistFileDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, output_dir, fold, database, split_idx, batch_size, train_test, need_jump):
        self.dataset = dataset
        self.output_dir = output_dir
        self.fold = fold
        self.database = database
        self.split_idx = split_idx
        self.batch_size = batch_size
        self.train_test =train_test
        self.need_jump = need_jump
        self.file_key_counter = defaultdict(int)

        self.sequence_name_list = self.dataset.sequence_name_list

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        if i > len(self):
            raise IndexError("Index too large , i = {}".format(i))
        img_path, from_img_path, AU_set, database_name =self.dataset.result_data[i]
        sequence_key = "_".join((img_path.split("/")[-3], img_path.split("/")[-2]))

        file_key_counter = self.file_key_counter[sequence_key] // self.batch_size + 1
        self.file_key_counter[sequence_key] += 1
        file_name = self.output_dir + os.sep + "{0}_{1}_fold_{2}".format(self.database, self.fold,
                                                                      self.split_idx) + "/{}".format(self.train_test)\
                    + os.sep + sequence_key + "@" + str(file_key_counter)+ ".npz"
        if os.path.exists(file_name) and self.need_jump:
            return None, None, None, img_path, file_key_counter
        dataset_return = list(self.dataset.get_example(i))
        dataset_return.append(file_key_counter)
        return tuple(dataset_return)