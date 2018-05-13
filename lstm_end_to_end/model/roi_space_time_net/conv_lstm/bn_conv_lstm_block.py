import chainer
import chainer.functions as F
import chainer.links as L
from lstm_end_to_end.model.roi_space_time_net.conv_lstm.seperate_batch_norm_1d import SeparatedBatchNorm1d

class BNConvLSTMCell(chainer.Chain):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, use_bias):
        """
          Initialize ConvLSTM cell.
          Parameters
          ----------
          input_size: (int, int)
              Height and width of input tensor as (height, width).
          input_dim: int
              Number of channels of input tensor.
          hidden_dim: int
              Number of channels of hidden state.
          kernel_size: (int, int)
              Size of the convolutional kernel.
          use_bias: bool
              Whether or not to add the bias.
        """
        super(BNConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        with self.init_scope():
            # if use_bias:
            #     self.bias = chainer.Parameter(shape=(4 * self.hidden_dim,))
            # else:
            #     self.add_persistent("bias", None)
            self.conv = L.Convolution2D(in_channels=self.input_dim + self.hidden_dim,
                                        out_channels=4 * self.hidden_dim,
                                        ksize=self.kernel_size, pad=self.padding,
                                        nobias=False)
            self.bn_ih = SeparatedBatchNorm1d(num_features=4 * self.hidden_dim, max_length=25)
            # self.bn_c = SeparatedBatchNorm1d(num_features=self.hidden_dim, max_length=25)

    def reset_parameters(self):
        self.bn_ih.reset_parameters()
        # self.bn_c.reset_parameters()
        self.bn_ih.beta.data[...] = 0.0
        self.bn_ih.gamma.data[...] = 0.1
        # self.bn_c.gamma.data[...] = 0.1

    def __call__(self, input_tensor, cur_state, time):
        h_cur, c_cur = cur_state
        # batch_size = h_cur.shape[0]
        combined = F.concat([input_tensor, h_cur], axis=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        combined_conv = self.bn_ih(combined_conv, time=time)
        assert combined_conv.shape[1] % self.hidden_dim == 0
        assert combined_conv.shape[1]//self.hidden_dim == 4
        cc_i, cc_f, cc_o, cc_g = F.split_axis(combined_conv, combined_conv.shape[1]//self.hidden_dim, axis=1)
        i = F.sigmoid(cc_i)
        f = F.sigmoid(cc_f)
        o = F.sigmoid(cc_o)
        g = F.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * F.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size):
        return chainer.Variable(self.xp.zeros((batch_size, self.hidden_dim, self.height, self.width), dtype=self.xp.float32)), \
            chainer.Variable(self.xp.zeros((batch_size, self.hidden_dim, self.height, self.width), dtype=self.xp.float32))


class BNConvLSTM(chainer.Chain):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(BNConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.cell_list = []

        with self.init_scope():
            for i in range(self.num_layers):
                cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
                setattr(self, "cell_{}".format(i), BNConvLSTMCell(input_size=(self.height, self.width),
                                                                input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i],
                                                                kernel_size=self.kernel_size[i], use_bias=self.bias))
                self.cell_list.append("cell_{}".format(i))
                getattr(self, "cell_{}".format(i)).reset_parameters()


    def __call__(self, input_tensor, hidden_state=None):
        """
            Parameters
            ----------
            input_tensor: todo
                5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
            hidden_state: todo
                None. todo implement stateful
            Returns
            -------
            last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = F.transpose(input_tensor, (1, 0, 2, 3, 4))
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.shape[0])
        layer_output_list = []
        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = getattr(self, self.cell_list[layer_idx])(input_tensor=cur_layer_input[:, t, :, :, :],
                                                                cur_state=[h, c], time=t)
                output_inner.append(h) # h is shape = B, C, H, W
            layer_output = F.stack(output_inner, axis=1)  # B, T, C, H, W, where T = seq_len
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
        if not self.return_all_layers:  # only return last layer of (B, T, C, H, W)
            layer_output_list = layer_output_list[-1:]

        return layer_output_list[0]

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(getattr(self, self.cell_list[i]).init_hidden(batch_size))
        return init_states


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param