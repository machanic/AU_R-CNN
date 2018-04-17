import chainer
import chainer.links as L
import chainer.functions as F


class ConvLSTMCell(chainer.Chain):
    def __init__(self, group_num, input_size, input_dim, hidden_dim, kernel_size, bias):
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
          bias: bool
              Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.group_num = group_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels=group_num * (self.input_dim + self.hidden_dim),
                                        out_channels=group_num * 4 * self.hidden_dim,
                                        ksize=self.kernel_size, pad=self.padding,
                                        nobias=not self.bias, groups=group_num)


    def __call__(self, input_tensor, cur_state):  # input_tensor and cur_state is B,F,C,H,W
        h_cur, c_cur = cur_state   # B, F, C, H, W
        mini_batch, frame_box_num, channel, height, width = input_tensor.shape
        combined = F.concat([input_tensor, h_cur], axis=2)  # concatenate along channel axis
        assert frame_box_num == self.group_num
        combined = F.reshape(combined, shape=(mini_batch, frame_box_num * combined.shape[2], height, width))
        conv_output = self.conv(combined)

        all_combined = list(F.split_axis(conv_output, self.group_num, axis=1, force_tuple=True))  # list(F) of B, C, H, W
        h_next_list = []
        c_next_list = []
        for f_idx, combined_conv in enumerate(all_combined):
            assert combined_conv.shape[1] % self.hidden_dim == 0
            assert combined_conv.shape[1]//self.hidden_dim == 4
            cc_i, cc_f, cc_o, cc_g = F.split_axis(combined_conv, combined_conv.shape[1]//self.hidden_dim, axis=1)
            i = F.sigmoid(cc_i)
            f = F.sigmoid(cc_f)
            o = F.sigmoid(cc_o)
            g = F.tanh(cc_g)
            c_next = f * c_cur[:, f_idx, :, :, : ] + i * g
            h_next = o * F.tanh(c_next)
            h_next_list.append(h_next)
            c_next_list.append(c_next)
        h_next_list = F.stack(h_next_list, axis=1)  # B, F, C, H, W
        c_next_list = F.stack(c_next_list, axis=1)  # B, F, C, H, W
        return h_next_list, c_next_list

    def init_hidden(self, batch_size):
        return chainer.Variable(self.xp.zeros((batch_size, self.group_num, self.hidden_dim, self.height, self.width),
                                              dtype=self.xp.float32)), \
            chainer.Variable(self.xp.zeros((batch_size, self.group_num,
                                            self.hidden_dim, self.height, self.width), dtype=self.xp.float32))


class ConvLSTM(chainer.Chain):
    def __init__(self, input_size, group_num, input_dim, hidden_dim, kernel_size, num_layers,
                  bias=True):
        super(ConvLSTM, self).__init__()
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
        self.bias = bias

        self.cell_list = []

        with self.init_scope():
            for i in range(self.num_layers):
                cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
                setattr(self, "cell_{}".format(i), ConvLSTMCell(input_size=(self.height, self.width), group_num=group_num,
                                                                input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i],
                                                                kernel_size=self.kernel_size[i], bias=self.bias))
                self.cell_list.append("cell_{}".format(i))

    def __call__(self, input_tensor, hidden_state=None):
        """
            Parameters
            ----------
            input_tensor: todo
                6-D Tensor either of shape (B, F, T, C, H, W)
            hidden_state: todo
                None. todo implement stateful
            Returns
            -------
            last_state_list, layer_output
        """
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.shape[0])
        seq_len = input_tensor.shape[2]
        cur_layer_input = input_tensor
        layer_output = None
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx] # B, F, C, H, W
            output_inner = []
            for t in range(seq_len):
                h, c = getattr(self, self.cell_list[layer_idx])(input_tensor=cur_layer_input[:, :, t, :, :, :],
                                                                cur_state=[h,c])   # B, F, C, H, W
                output_inner.append(h) # h is shape = B, F, C, H, W
            layer_output = F.stack(output_inner, axis=2)  # B, F, T, C, H, W, where T = seq_len
            cur_layer_input = layer_output

        return layer_output

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