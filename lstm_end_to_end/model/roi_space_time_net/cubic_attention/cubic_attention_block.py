import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

linear_init = chainer.initializers.LeCunUniform()




class ScaledDotProductAttention(chainer.Chain):
    def __init__(self, d_k, height, width, attn_dropout=0.1, use_linear_project=False):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_dim = 2048
        self.temper = d_k ** 0.5
        self.attn_dropout = attn_dropout
        self.use_linear_project = use_linear_project
        if use_linear_project:
            self.temper = self.fc_dim ** 0.5
            with self.init_scope():
                self.fc_matmul_q = L.Linear(d_k * height * width, self.fc_dim)
                self.fc_matmul_k = L.Linear(d_k * height * width, self.fc_dim)

    def __call__(self, q, k, v, attn_mask=None):
        # the intention: to calculate the similarity of len_q of C_k x H_k x W_k and len_k of C_k x H_k x W_k
        # q k v shape = (n_head* batch) x len_q/len_k x C_k x H_k x W_k or C_v x H_v x W_v
        # after = (n_head*batch) x len_q x len_k

        n_head_batch, len_q, C_q, H_q, W_q = q.shape
        n_head_batch, len_k, C_k, H_k, W_k = k.shape
        if self.use_linear_project:
            q = F.reshape(q, shape=(n_head_batch * len_q, -1))
            k = F.reshape(k, shape=(n_head_batch * len_k, -1))
            q = self.fc_matmul_q(q)
            k = self.fc_matmul_k(k)
            q = F.reshape(q, shape=(n_head_batch, len_q, self.fc_dim))
            k = F.reshape(k, shape=(n_head_batch, len_k, self.fc_dim))
            attn = F.matmul(q, F.transpose(k, axes=(0,2,1))) / self.temper  #
            # (n_head*batch) of matrix multiply (len_q x d_q) x (d_k x len_k) = (n_head*batch) x len_q x len_k

            if attn_mask is not None:
                assert attn_mask.shape == attn.shape, \
                    'Attention mask shape {0} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{1}.'.format(attn_mask.shape, attn.shape)
                if hasattr(attn_mask, "data"):
                    attn_mask.data = attn_mask.data.astype(bool)
                attn = F.where(attn_mask, self.xp.full(attn.shape, -np.inf, 'f'), attn)
            attn = F.softmax(attn, axis=2)  # (n_head*batch) x len_q x len_k
            attn = F.dropout(attn, ratio=self.attn_dropout)
            n_head_batch, len_v, C_v, H_v, W_v = v.shape
            v = F.reshape(v, shape=(n_head_batch, len_k, -1))
            output = F.matmul(attn, v) #  (n_head*batch) matrix of (len_q x len_k) x (len_v x d_v) = (n_head*batch) x len_q x d_v
            output = F.reshape(output, shape=(n_head_batch, len_q, C_v, H_v, W_v))
            return output, attn
        else:
            # n_head_batch, len_q, C_q, H_q, W_q = q.shape
            # n_head_batch, len_k, , H_k, W_k = k.shape
            q = F.transpose(q, axes=(0, 3, 4, 1, 2))  # shape = n_head_batch, H_q, W_q, len_q, C_q
            k = F.transpose(k, axes=(0, 3, 4, 2, 1)) # shape = n_head_batch, H_k, W_k,  C_k, len_k
            attn = F.matmul(q, k) / self.temper # shape = n_head_batch, H_q, W_q, len_q, len_k
            # # (n_head_batch, H_q, W_q) batch of matrix multiply (len_q x d_k) x (d_k x len_k) = n_head_batch, H_q, W_q x len_q x len_k

            if attn_mask is not None:
                assert attn_mask.shape == attn.shape, \
                    'Attention mask shape {0} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{1}.'.format(attn_mask.shape, attn.shape)
                if hasattr(attn_mask, "data"):
                    attn_mask.data = attn_mask.data.astype(bool)
                attn = F.where(attn_mask, self.xp.full(attn.shape, -np.inf, 'f'), attn)
            attn = F.softmax(attn, axis=4)
            attn = F.dropout(attn, ratio=self.attn_dropout)
            # v shape = n_head_batch, len_v, C_v , H_v , W_v , note that len_k == len_v
            v = F.transpose(v, axes=(0, 3, 4, 1, 2))  # n_head_batch, H_v, W_v, len_v, C_v
            output = F.matmul(attn,
                              v)  # (n_head_batch, H_v, W_v) matrix of (len_q x len_k) x (len_v x C_v) = (n_head_batch, H_v, W_v) x len_q x C_v
            # 因为d_k == d_q，所以 output = (n_head*batch) x len_q x d_v
            output = F.transpose(output, (0, 3, 4, 1, 2))  # shape = n_head_batch, len_q, C_v, H_v, W_v
            attn = F.transpose(attn, (0, 3,4,1,2 )) # shape = n_head_batch, len_q, len_k, H_q, W_q
            return output, attn



class CubicMultiHeadAttention(chainer.Chain):

    def __init__(self, n_heads, d_model, height, width, d_k=64, d_v=64, dropout=0.1, bias=True, ksize=(3,3),
                 use_linear_project=False):
        super(CubicMultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout_ratio = dropout
        if isinstance(ksize, tuple):
            padding = ksize[0] // 2, ksize[1] // 2
        else:
            padding = ksize//2
        with self.init_scope():

            self.conv_qs = L.Convolution2D(in_channels=d_model, out_channels=d_k,
                                           ksize=ksize, pad=padding, nobias=not bias, initialW=linear_init)
            self.conv_ks = L.Convolution2D(in_channels=d_model, out_channels=d_k,
                                           ksize=ksize, pad=padding, nobias=not bias, initialW=linear_init)
            self.conv_vs = L.Convolution2D(in_channels=d_model, out_channels=d_v,
                                           ksize=ksize, pad=padding, nobias=not bias, initialW=linear_init)
            self.attention = ScaledDotProductAttention(d_k, height, width, dropout,use_linear_project)

            self.bn = L.BatchNormalization(d_model)


    def __call__(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_heads
        batch_size, len_q, d_model, height_q, width_q = q.shape
        batch_size, len_k, d_model, height_k, width_k = k.shape
        batch_size, len_v, d_model, height_v, width_v = v.shape
        assert q.shape[2] == k.shape[2] == v.shape[2]
        residual = q
        # treat as a (n_head) size batch, shape = (heads x batch), number_words, d_model; then (heads, (batch x len_q), d_model)
        q_s = F.tile(q, reps=(n_head, 1, 1, 1, 1)).reshape(n_head * batch_size * len_q, d_model, height_q, width_q)  # n_head x (batch_size*len_q) x d_model x height_q x width_q
        k_s = F.tile(k, reps=(n_head, 1, 1, 1, 1)).reshape(n_head * batch_size * len_k, d_model, height_k, width_k)  # n_head x (batch_size*len_k) x d_model x height_k x width_k
        v_s = F.tile(v, reps=(n_head, 1, 1, 1, 1)).reshape(n_head * batch_size * len_v, d_model, height_v, width_v)  # n_head x (batch_size*len_v) x d_model x height_v x width_v

        q_s = self.conv_qs(q_s)  # shape = (n_head * batch_size * len_q), d_q, height_q, width_q
        k_s = self.conv_ks(k_s)  # shape = (n_head * batch_size * len_k), d_k, height_k, width_k
        v_s = self.conv_vs(v_s)

        q_s = F.reshape(q_s, shape=(n_head * batch_size, len_q, d_k, height_q, width_q))
        k_s = F.reshape(k_s, shape=(n_head * batch_size, len_k, d_k, height_k, width_k))
        v_s = F.reshape(v_s, shape=(n_head * batch_size, len_v, d_v, height_v, width_v))

        # outputs size = (n_head * mb_size) x len_q x d_v, attns size = (n_head*mb_size) x len_q x len_k
        if attn_mask is not None:
            attn_mask = F.tile(attn_mask, reps=(n_head, 1, 1, 1, 1))
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask) # n_head * batch, len_q, d_v, H_v, W_v
        outputs = F.reshape(outputs, shape=(n_head, batch_size, len_q, d_v, height_v, width_v))
        outputs = F.transpose(outputs, axes=(1,2,0,3,4,5)) # batch_size, len_q ,n_head , d_v, height_v, width_v
        outputs = F.reshape(outputs, shape=(batch_size * len_q, n_head * d_v, height_v, width_v)) # n_head * d_v = d_model
        # outputs = self.proj(outputs)  # batch_size * len_q, d_model, height_v, width_v
        # project back to residual size
        outputs = F.dropout(outputs, self.dropout_ratio)
        residual = F.reshape(residual, shape=(batch_size * len_q, d_model, height_q, width_q))
        final_outputs = self.bn(outputs + residual)
        return final_outputs.reshape(batch_size, len_q, d_model, height_v, width_v)
