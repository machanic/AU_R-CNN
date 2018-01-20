import chainer

import chainer.functions as F
import chainer.links as L
import numpy as np

class GraphAttentionBlock(chainer.Chain):

    def __init__(self, input_node_dim, F_, attn_heads=1, attn_heads_reduction='concat',
                 attn_dropout=0.5, activation='relu', frame_node_num=9):

        super(GraphAttentionBlock, self).__init__()
        self.frame_node_num = frame_node_num
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
                convert_dim_lstm = L.NStepLSTM(1, input_node_dim, self.F_, dropout=0.0)
                setattr(self, "dim_lstm_{}".format(head), convert_dim_lstm)
                self.kernels.append(convert_dim_lstm)
                attn_lstm = L.NStepLSTM(1, 2 * self.F_, 1, dropout=0.0)
                setattr(self, "attn_lstm_{}".format(head), attn_lstm)
                self.attn_kernels.append(attn_lstm)

    def __call__(self, X):
        outputs = []
        X = X.reshape(-1, self.frame_node_num, X.shape[-1])  # Frame x Box x input_node_dim
        frame_num = X.shape[0]
        for head in range(self.attn_heads):
            kernel_lstm = self.kernels[head]  # convert node dimension: input list of Frame x F, output list of Frame x F'
            attn_lstm = self.attn_kernels[head]   # attention kernel: input list of Frame x 2F' ,output list of Frame x 1
            # Compute inputs to attention network
            X_T = F.transpose(X, (1,0,2)) # Box x Frame x input_node_dim
            X_list = [e for e in X_T] # list of Frame x F
            _, _, linear_transfer_X = kernel_lstm(None, None, X_list)   # list(Box) of (Frame x F')
            # Compute feature combinations
            linear_transfer_X = F.stack(linear_transfer_X) # Box x Frame x F'
            linear_transfer_X_T = F.transpose(linear_transfer_X, (1,0,2)) # Frame x Box x F'
            # after tile: Frame x Box x (Box x F'), then Frame x Box^2 x F'
            repeated = F.reshape(F.tile(linear_transfer_X_T, (1,1, self.frame_node_num)), (frame_num,
                                                                self.frame_node_num * self.frame_node_num, self.F_))
            tiled = F.tile(linear_transfer_X_T, (1, self.frame_node_num, 1)) # Frame x Box^2 x F'
            combinations = F.concat([repeated, tiled], axis=2)  # (Frame x Box^2 x 2F') # this will be all combinations Box x Box include self to self

            combination_slices = F.reshape(combinations, (frame_num, self.frame_node_num, self.frame_node_num, 2 * self.F_)) # Frame x Box x Box x 2F'
            # Attention head
            combination_slices_T = F.transpose(combination_slices, (1,2,0,3)) # Box x Box x Frame x 2F'
            combination_slices_T = combination_slices_T.reshape(self.frame_node_num * self.frame_node_num, frame_num, 2 * self.F_) # Box^2 x Frame x 2F'
            combination_slices_list = [e for e in combination_slices_T]  # list(Box^2) of Frame x 2F' variable
            _,_, attn_result_list = attn_lstm(None,None, combination_slices_list)  #FIXME may bug, because it is list pass in, softmax works? # list(Box^2) of Frame x 1   # a(Wh_i, Wh_j) in the paper (N x N x 1),
            attn_result = F.stack(attn_result_list) # Box^2 x Frame x 1
            dense = F.squeeze(attn_result)  # Box^2 x Frame (then squeeze to remove last 1)
            dense = F.transpose(dense.reshape(self.frame_node_num, self.frame_node_num, frame_num), (2,0,1)) # Frame x Box x Box
            # Feed masked values to softmax
            softmax_val = F.softmax(dense,axis=2)  # paper eqn.3 alpha
            dropout_val = F.dropout(softmax_val, ratio=self.attn_dropout)   # Frame x Box x Box
            # Linear combination with neighbors' features
            node_features = F.batch_matmul(dropout_val, linear_transfer_X_T)  # (Frame x Box x Box) x (Frame x Box x F') = Frame x Box x F'
            node_features = node_features.reshape(frame_num * self.frame_node_num, self.F_)  # shape = N x F', where N = Frame x Box
            if self.attn_heads_reduction == 'concat' and self.activation is not None:
                # In case of 'concat', we compute the activation here (Eq 5)
                node_features = self.activation(node_features)  # shape = N x F', where N = Frame x Box
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
    def __init__(self, input_dim, hidden_dim, class_number, atten_heads, layers_num, frame_node_num):
        super(GraphAttentionModel, self).__init__()
        input_size = input_dim
        self.layers_num = layers_num
        with self.init_scope():
            for i in range(layers_num):
                if i != layers_num-1:

                    setattr(self, "attn_{}".format(i), GraphAttentionBlock(input_size, hidden_dim,
                                                                           atten_heads, 'concat', 0.5, 'relu',frame_node_num))
                    input_size = atten_heads * hidden_dim
                else: # last layer
                    setattr(self, "attn_{}".format(i), GraphAttentionBlock(atten_heads * hidden_dim, class_number,
                                                                           atten_heads, 'average', 0.5, 'relu',frame_node_num))


    def get_gt_label_one_graph(self, xp, crf_pact_structure, is_bin=True):
        sample = crf_pact_structure.sample
        if not is_bin:
            node_label_one_video = xp.zeros(shape=len(sample.node_list), dtype=xp.int32)
        else:
            node_label_one_video = xp.zeros(shape=(len(sample.node_list), sample.label_bin_len), dtype=xp.int32)
        for idx, node in enumerate(sample.node_list):
            assert node.id == idx
            if is_bin:
                label_bin = xp.asarray(node.label_bin)
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
            gt_label = chainer.Variable(self.get_gt_label_one_graph(xp, crf_pact_structure, is_bin=True))  # N x Y
            for i in range(self.layers_num):
                X = getattr(self, "attn_{}".format(i))(X)
            loss += F.sigmoid_cross_entropy(X, gt_label)
            accuracy = F.binary_accuracy(X, gt_label)
        chainer.reporter.report({"loss":loss, "accuracy":accuracy},self)
        return loss