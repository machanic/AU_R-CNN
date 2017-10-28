from collections import defaultdict

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import initializers

from structural_rnn.model.open_crf.cython.factor_graph import FactorGraph
import random
'''
nodeRNNs[nm] = [multilayerLSTM(LSTMs,skip_input=True,skip_output=True,input_output_fused=True),
				FCLayer('rectify',args.fc_init,size=args.fc_size,rng=rng),
				FCLayer('rectify',args.fc_init,size=100,rng=rng),
				FCLayer('linear',args.fc_init,size=num_classes,rng=rng)
				]
				
edgeRNNs[em] = [TemporalInputFeatures(nodeFeatureLength[nm]),
				#AddNoiseToInput(rng=rng),
				FCLayer('rectify',args.fc_init,size=args.fc_size,rng=rng),
				FCLayer('linear',args.fc_init,size=args.fc_size,rng=rng),
				multilayerLSTM(LSTMs_edge,skip_input=True,skip_output=True,input_output_fused=True)
				]
'''


class NodeRNN(chainer.Chain):

    def __init__(self, insize, outsize, initialW=None):
        super(NodeRNN, self).__init__()
        if not initialW:
            initialW = initializers.HeNormal()
        self.n_layer = 1
        with self.init_scope():

            self.lstm1 = L.NStepLSTM(self.n_layer, insize, 512, dropout=0) #dropout = 0.0
            self.fc2 = L.Linear(512, 256, initialW=initialW)
            self.fc3 = L.Linear(256, 100, initialW=initialW)
            self.fc4 = L.Linear(100, outsize, initialW=initialW)

    def __call__(self, xs):
        xp = chainer.cuda.cupy.get_array_module(xs[0].data)
        hx = None
        cx = None
        # with chainer.no_backprop_mode():
        #     hx = chainer.Variable(xp.zeros((self.n_layer, len(xs), 512), dtype=xp.float32))
        #     cx = chainer.Variable(xp.zeros((self.n_layer, len(xs), 512), dtype=xp.float32))
        _, _, hs = self.lstm1(hx, cx, xs)  # hs is list of T x D variable
        # hs = [F.dropout(h) for h in hs]
        hs = [F.relu(self.fc2(h)) for h in hs]
        hs = [F.relu(self.fc3(h)) for h in hs]
        return [self.fc4(h) for h in hs]


class EdgeRNN(chainer.Chain):

    def __init__(self, insize, outsize, initialW=None):
        super(EdgeRNN, self).__init__()
        self.n_layer = 1
        self.outsize = outsize
        if not initialW:
            initialW = initializers.HeNormal()

        with self.init_scope():
            self.fc1 = L.Linear(insize, 256, initialW=initialW)
            self.fc2 = L.Linear(256, 256, initialW=initialW)
            self.lstm3 = L.NStepLSTM(self.n_layer, 256, outsize, dropout=0.0)  #dropout = 0.0

    def __call__(self, xs):
        xp = chainer.cuda.cupy.get_array_module(xs[0].data)
        hs = [F.relu(self.fc1(x)) for x in xs]
        hs = [F.relu(self.fc2(h)) for h in hs]
        hx = None
        cx = None
        # hx = chainer.Variable(xp.zeros((self.n_layer, len(xs), self.outsize), dtype=xp.float32))
        # cx = chainer.Variable(xp.zeros((self.n_layer, len(xs), self.outsize), dtype=xp.float32))

        _, _, hs = self.lstm3(hx, cx, hs)
        # https://docs.chainer.org/en/stable/reference/core/configuration.html?highlight=config and https://stackoverflow.com/questions/45757330/how-to-use-chainer-using-config-to-stop-f-dropout-in-evaluate-predict-process-in
        # hs = [F.dropout(h) for h in hs]
        return hs


class StructuralRNN(chainer.Chain):

    # 一共2层，底下平行的edge层（一个是feature直通的FC层变换维度， 另一个是edge层）
    #  上面一层是node层，输入来自与他相连的node层和与相连的edge层，每个组件都是一个component, 类似ResNet的Block
    #  参数G是描述性的Graph，可以取第一帧Graph，但切不可传入整个video的Graph，而out_size是输出的node outsize可以根据是否叠加OpenCRF来定
    # in_size 指的是每个node的input feature length
    def __init__(self, G:FactorGraph,  in_size, out_size, initialW=None):

        super(StructuralRNN, self).__init__()
        self.out_size = out_size
        self.in_size = in_size

        self.node_id_neighbor = defaultdict(list)  # 现在需要造出每个node_id(RNN的node_id)对应的跨越frame的node_list
        if not initialW:
            initialW = initializers.HeNormal()
        with self.init_scope():
            self.node_feature_convert_len = 512
            self.node_feature_convert_fc = L.Linear(in_size, self.node_feature_convert_len, initialW=initialW)

            self.bottom = dict() # key is ",".join(box_id_a,box_id_b)
            self.top = dict() # key is box_id
            # build bottom layer EdgeRNN first
            for factor_node in G.factor_node:
                neighbors = factor_node.neighbor
                var_node_a = neighbors[0]
                var_node_b = neighbors[1]
                feature_len = 2 * in_size
                edge_RNN_id = ",".join(map(str, sorted([int(var_node_a.id), int(var_node_b.id)])))
                self.add_link("EdgeRNN_{}".format(edge_RNN_id), EdgeRNN(insize=feature_len, outsize=self.node_feature_convert_len))
                self.bottom[edge_RNN_id] = getattr(self, "EdgeRNN_{}".format(edge_RNN_id))  # 输出是node feature, 最后node feature concat起来
            # build top layer NodeRNN
            for node in G.var_node:
                node_id = node.id
                neighbors = node.neighbor
                for factor_node in neighbors:
                    can_add = False
                    for var_node in factor_node.neighbor:
                        if var_node.id == node_id and not can_add:
                            can_add = True
                            continue
                        self.node_id_neighbor[node_id].append(var_node.id)  # 这样应该也会将自己对自己相连的id包含在内
                for key, val_list in self.node_id_neighbor.items():
                    self.node_id_neighbor[key] = sorted(val_list)
                feature_len = self.node_feature_convert_len * (len(neighbors) + 1)  # concat edgeRNN out and node feature(after convert dim)
                self.add_link("NodeRNN_{}".format(node_id), NodeRNN(insize=feature_len, outsize=out_size) )
                self.top[str(node_id)] = getattr(self, "NodeRNN_{}".format(node_id))

    def predict(self, x, crf_pact_structure, is_bin=False):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        xp = chainer.cuda.cupy.get_array_module(x)

        with chainer.no_backprop_mode():
            xs = F.expand_dims(x, 0)
            crf_pact_structures = [crf_pact_structure]
            xs = self.__call__(xs, crf_pact_structures)
            xs = F.copy(xs, -1)
            pred_score_array = xs.data[0]
            pred = np.argmax(pred_score_array, axis=1)
            assert len(pred) == x.shape[0]
        return pred.astype(np.int32)  # return N x 1, where N is number of nodes


    def __call__(self, xs, crf_pact_structures):  # xs is chainer.Variable
        '''
        只支持batch_size = 1
        传入NodeRNN的是两部分: x: 表示node的feature, 而edge的feature来自于node的feature的concat
        x的shape= (T, in_size) T表示nodeid编号，in_size表示每个node自带的的feature的length
        详见例子：https://github.com/kei-s/chainer-ptb-nsteplstm/blob/master/train_ptb_nstep.py#L24
        :param boxid_features: key= box_id, value= feature_list
        :return : chainer.Variable shape= B * N * D , B is batch_size, N is one video all nodes count, D is each node output vector
        '''

        xp = chainer.cuda.get_array_module(xs.data)
        for idx, x in enumerate(xs): # xs is shape B x N x D. B is batch_size, always = 1
            x = x.reshape(-1, self.in_size)  # because of padding
            crf_pact_structure = crf_pact_structures[idx]
            # node_id_convert = crf_pact_structure.node_id_convert # this long sequence node_id convert => root node_id (NodeRNN id)
            nodeRNN_id_dict = crf_pact_structure.nodeRNN_id_dict

            edge_out_dict = dict()
            for edge_RNN_id, edge_RNN in self.bottom.items():
                # print("edgeRNN:{}".format(edge_RNN_id))
                orig_node_id_a, orig_node_id_b = edge_RNN_id.split(",")
                node_list_a = nodeRNN_id_dict[int(orig_node_id_a)]
                node_list_b = nodeRNN_id_dict[int(orig_node_id_b)]
                assert len(node_list_a) == len(node_list_b)
                fetch_x_index = np.array(list(zip(node_list_a, node_list_b)))  # T x 2
                edge_feature = x[fetch_x_index.flatten(), :]  # (T x 2) x D
                edge_feature = edge_feature.reshape(fetch_x_index.shape[0], 2 * x.shape[-1])  # x shape N x D, fetch_x_index shape = T x 2,
                edge_out_dict[edge_RNN_id] = edge_RNN([edge_feature])[0]

            node_output = []
            for node_RNN_id, node_RNN in sorted(self.top.items(), key=lambda e:int(e[0])):
                # print("nodeRNN:{}".format(node_RNN_id))
                neighbor_id_list = self.node_id_neighbor[int(node_RNN_id)]
                concat_features = []
                nodeid_list = nodeRNN_id_dict[int(node_RNN_id)]  # length = T
                converted_node_feature = self.node_feature_convert_fc(x[nodeid_list])  # shape = T x in_size
                concat_features.append(converted_node_feature)
                for neighbor_id in neighbor_id_list:
                    edge_RNN_id = ",".join(map(str, sorted([int(node_RNN_id), int(neighbor_id)])))
                    edge_out = edge_out_dict[edge_RNN_id]
                    concat_features.append(edge_out)  # list of  T x 256 shape
                frame_num = len(nodeid_list)

                concat_features = F.stack(concat_features, axis=0)  # shape = (neighbor + 1) x T x 256
                assert concat_features.shape[2] == self.node_feature_convert_len
                concat_features = F.transpose(concat_features, (1, 0, 2))  # shape = T x (neighbor + 1) x 256
                concat_features = F.reshape(concat_features, (frame_num, -1))  # shape = T x ((neighbor + 1) x 256)
                node_output.append(node_RNN([concat_features])[0])  # output= list of T x out_size
            node_output = F.stack(node_output)  # shape nodeRNN_num x T x out_size
            node_output = F.transpose(node_output,axes=(1,0,2))  # reorder
            node_output = node_output.reshape(-1, self.out_size)  # shape N x out_size
            # reorder to nodeid list order
            # time_used = {int(node_RNN_id):0 for node_RNN_id in self.top.keys()}
            # one_video_nodes_output = list()
            # for i in range(x.shape[0]):
            #     node_id = i  # node_id 来自于factor_graph里的var_node.id，从0开始
            #     node_RNN_id = int(node_id_convert[node_id])
            #     one_video_nodes_output.append(node_output_dict[node_RNN_id][time_used[node_RNN_id]]) # 此处在variable下标索引，是否可以反传一定要保证nodeid从小到大是沿着frame顺序的
            #     time_used[node_RNN_id] += 1
            # nodes_output.append(F.stack(one_video_nodes_output, axis=0))
        return F.expand_dims(node_output,axis=0)  # return shape B x N x D. B is batch_size,  but can only deal with one, N is number of variable nodes in graph D is out_size
