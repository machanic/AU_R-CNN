
import copy

class DataSet(object):

    def __init__(self, data_builder, split_test_size):
        self.data_builder = data_builder
        self.data_builder.read()  # read!!
        self.data_builder_trn = copy.deepcopy(self.data_builder)
        self.data_builder_test = copy.deepcopy(self.data_builder)
        self.data_builder_trn.images = self.data_builder.images[0:-split_test_size]
        self.data_builder_trn.labels = self.data_builder.labels[0:-split_test_size]
        self.data_builder_test.images = self.data_builder.images[-split_test_size:]
        self.data_builder_test.labels = self.data_builder.labels[-split_test_size:]
        self.train = self.data_builder_trn
        self.train._num_examples = self.train._num_examples - split_test_size
        self.test = self.data_builder_test
        self.test._num_examples = split_test_size

def generate_from_imgs(dataset, num_classes):
    while True:
        imgs, labels = dataset.next_batch(10)
        yield imgs, np_utils.to_categorical(labels, num_classes)