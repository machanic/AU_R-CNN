import chainer

import chainer.functions as F
import chainer.links as L
import numpy as np

class GraphAttentionBlock(chainer.Chain):

    def __init__(self, input_node_dim, F_, attn_heads=1, attn_heads_reduction='concat',
                 attn_dropout=0.5, activation='relu'):
        super(GraphAttentionBlock, self).__init__()
        activations = {"relu" : F.relu, "elu": F.elu}
        self.F_ = F_  #  Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction   # 'concat' or 'average' (Eq 5 and 6 in the paper)
        self.attn_dropout = attn_dropout  # Internal dropout rate for attention coefficients
        self.activation = activations[activation] # Optional nonlinearity (Eq 4 in the paper)
        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.attn_kernels = []  #  Attention kernels for attention heads
        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_
        with self.init_scope():
            for head in range(self.attn_heads):
                convert_dim_fc = L.Linear(input_node_dim, self.F_)
                setattr(self, "dim_fc_{}".format(head), convert_dim_fc)
                self.kernels.append(convert_dim_fc)
                attn_fc = L.Linear(2 * self.F_, 1)
                setattr(self, "attn_fc_{}".format(head), attn_fc)
                self.attn_kernels.append(attn_fc)

    def __call__(self, X, A):  # A is Adjacency Matrix
        N = X.shape[0]
        outputs = []

        for head in range(self.attn_heads):
            kernel_fc = self.kernels[head]  # W in paper (F x F')
            attn_fc = self.attn_kernels[head]   # Attention kernel a in the paper (2F' x 1)
            # Compute inputs to attention network
            linear_transfer_X = kernel_fc(X)   # (N x F')
            # Compute feature combinations
            linear_transfer_X = F.tile(linear_transfer_X, (1,N))
            repeated = F.reshape(linear_transfer_X, (N * N, self.F_))  # after tile: N x (F' x N), then N^2 x F'
            tiled = F.tile(linear_transfer_X, (N, 1)) # (N^2 x F')
            combinations = F.concat([repeated, tiled], axis=1)  # (N^2 x 2F') # this will be all combinations N x N include self to self
            # Attention head
            dense = F.squeeze(attn_fc(combinations)) # a(Wh_i, Wh_j) in the paper (N^2 x 1), then squeeze to remove last 1
            dense = dense.reshape(N, N)
            dense = F.leaky_relu(dense)
            # Mask values before activation (Vaswani et al., 2017)
            comparison = (A == 0)  # true or false of each element
            mask = F.where(comparison, np.ones_like(A) * -10e9, np.zeros_like(A)) # if A ==0, choose -10e9, else choose 0
            # this mask elements: if A == 0: -10e9, if A!=0: 0
            masked = dense + mask  # push non-neighbor elements to -10e9, shape = N x N
            # Feed masked values to softmax
            softmax_val = F.softmax(masked,axis=1)  # paper eqn.3 alpha,  push non-neighbor node value to almost 0
            dropout_val = F.dropout(softmax_val, ratio=self.attn_dropout)  # shape = N x N
            # Linear combination with neighbors' features
            node_features = F.matmul(dropout_val, linear_transfer_X)  # (N x N) x (N x F') = N x F'
            if self.attn_heads_reduction == 'concat' and self.activation is not None:
                # In case of 'concat', we compute the activation here (Eq 5)
                node_features = self.activation(node_features)  # shape = N x F'
            # Add output of attention head to final output
            outputs.append(node_features) # add one head output
        # Reduce the attention heads output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = F.concat(outputs, axis=1) # shape = N x KF' , where K is head count
        else:
            output = F.mean(F.stack(outputs), axis=0) #shape = N x F'
            if self.activation is not None:
                output = self.activation(output)
        return output


class GraphAttentionModel(chainer.Chain):

    # note that class_number is binary length
    def __init__(self, input_dim, hidden_dim, class_number, atten_heads, layers_num):
        super(GraphAttentionModel, self).__init__()
        input_size = input_dim
        self.layers_num = layers_num
        with self.init_scope():
            for i in range(layers_num):
                if i != layers_num-1:
                    setattr(self, "attn_{}".format(i), GraphAttentionBlock(input_size, hidden_dim, atten_heads, 'concat', 0.5, 'elu'))
                    input_size = hidden_dim
                else: # last layer
                    setattr(self, "attn_{}".format(i), GraphAttentionBlock(hidden_dim, class_number, atten_heads, 'average', 0.5, 'elu'))


    def get_gt_label_one_graph(self, xp, crf_pact_structure, is_bin=True):
        sample = crf_pact_structure.sample
        if not is_bin:
            node_label_one_video = xp.zeros(shape=len(sample.node_list), dtype=xp.int32)
        else:
            node_label_one_video = xp.zeros(shape=(len(sample.node_list), sample.label_bin_len), dtype=xp.int32)
        for idx, node in enumerate(sample.node_list):
            assert node.id == idx
            if is_bin:
                label_bin = node.label_bin
                node_label_one_video[node.id] = label_bin
            else:
                label = node.label
                node_label_one_video[node.id] = label
        return node_label_one_video


    def predict(self, X, crf_pact_structure):
        with chainer.no_backprop_mode():
            if not isinstance(X, chainer.Variable):
                X = chainer.Variable(X)
            xp = chainer.cuda.get_array_module(X)
            A = crf_pact_structure.A
            for i in range(self.layers_num):
                X = getattr(self, "attn_{}".format(i))(X, A)
            pred_score = F.copy(X, -1)
            pred_score = pred_score.data
            pred_score = pred_score > 0
            pred_score = pred_score.astype(np.int32)
        return pred_score

    def __call__(self, xs, crf_pact_structures):
        loss = 0.0
        xp = chainer.cuda.cupy.get_array_module(xs.data)
        accuracy = 0.0
        for idx, X in enumerate(xs):
            crf_pact_structure = crf_pact_structures[idx]
            gt_label = self.get_gt_label_one_graph(xp, crf_pact_structure, is_bin=True)  # N x Y
            A = crf_pact_structure.A
            for i in range(self.layers_num):
                X = getattr(self, "attn_{}".format(i))(X, A)
            loss += F.sigmoid_cross_entropy(X, gt_label)
            accuracy = F.binary_accuracy(X, gt_label)
        chainer.reporter.report({"loss":loss, "accuracy":accuracy})
        return loss