import numpy as np
import tensorflow as tf

from network.nn import leaky_relu, softsample_word, argmax_word

class BaseModel(object):
    def __init__(self, args, vocab):
        self.dim_emb = args.dim_emb
        self.vocab_size = vocab.size
        self.dim_y = args.dim_y
        self.dim_z = args.dim_z
        self.dim_h = self.dim_y + self.dim_z
        self.max_len = args.max_len
        self.dropout_rate = args.dropout_rate
        self.learning_rate = args.learning_rate
        self.rho = args.rho
        self.gamma = args.gamma_init

        self.pretrain_epochs = args.pretrain_epochs

        # initializer for word embeeding
        initializer = np.random.random_sample((self.vocab_size, self.dim_emb)) - 0.5
        self.word_init = initializer.astype(np.float32)


    def build_placeholder(self):
        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.batch_len = tf.placeholder(tf.int32,
            name='batch_len')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None],    #size * len
            name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='dec_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.dec_mask = tf.placeholder(tf.float32, [None, None],
            name='dec_mask')
        self.labels = tf.placeholder(tf.float32, [None],
            name='labels')
        self.enc_lens = tf.placeholder(tf.float32, [None],
            name='enc_lens')

    def create_cell(self, dim, n_layers, dropout, scope=None):
        with tf.variable_scope(scope or "rnn"):
            cell = tf.nn.rnn_cell.GRUCell(dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                input_keep_prob=dropout)
            if n_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layers)
        return cell

    def create_cell_with_dims(self, args, hidden_dim, input_dim, dropout, scope):
        cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
        inputs = tf.placeholder(tf.float32, [args.batch_size, args.max_len, input_dim])
        with tf.variable_scope(scope):
            _, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        return cell

    def linear(self, inp, dim_out, scope, reuse=False):
        dim_in = inp.get_shape().as_list()[-1]
        with tf.variable_scope(scope) as vs:
            if reuse:
                vs.reuse_variables()
            W = tf.get_variable('W', [dim_in, dim_out])
            b = tf.get_variable('b', [dim_out])
        return tf.matmul(inp, W) + b

    def rnn_decode(self, h, inp, length, cell, loop_func, scope):
        h_seq, output_ids = [], []

        with tf.variable_scope(scope):
                tf.get_variable_scope().reuse_variables()
                for t in range(length):
                    h_seq.append(tf.expand_dims(h, 1))
                    output, h = cell(inp, h)
                    inp, ids = loop_func(output)
                    output_ids.append(tf.expand_dims(ids, 1))
                h_seq.append(tf.expand_dims(h, 1))

        return tf.concat(h_seq, 1), tf.concat(output_ids, 1)

    def run_decoder(self, decoder, dec_inputs, embedding, projection, origin_info, transfer_info):
        go = dec_inputs[:,0,:]
        soft_func = softsample_word(self.dropout, projection['W'], projection['b'], embedding,
            self.gamma)
        hard_func = argmax_word(self.dropout, projection['W'], projection['b'], embedding)


        soft_tsf_hiddens, soft_tsf_ids, = self.rnn_decode(
            transfer_info, go, self.max_len, decoder, soft_func, scope='decoder')

        _, rec_ids = self.rnn_decode(
            origin_info, go, self.max_len, decoder, hard_func, scope='decoder')
        _, tsf_ids = self.rnn_decode(
            transfer_info, go, self.max_len, decoder, hard_func, scope='decoder')
        return soft_tsf_hiddens, soft_tsf_ids, rec_ids, tsf_ids

    def cnn(self, inp, filter_sizes, n_filters, dropout, scope, reuse=False):
        dim = inp.get_shape().as_list()[-1]
        inp = tf.expand_dims(inp, -1)

        with tf.variable_scope(scope) as vs:
            if reuse:
                vs.reuse_variables()

            outputs = []
            for size in filter_sizes:
                with tf.variable_scope('conv-maxpool-%s' % size):
                    W = tf.get_variable('W', [size, dim, 1, n_filters])
                    b = tf.get_variable('b', [n_filters])
                    conv = tf.nn.conv2d(inp, W,
                        strides=[1, 1, 1, 1], padding='VALID')
                    h = leaky_relu(conv + b)
                    # max pooling over time
                    pooled = tf.reduce_max(h, reduction_indices=1)
                    pooled = tf.reshape(pooled, [-1, n_filters])
                    outputs.append(pooled)
            outputs = tf.concat(outputs, 1)
            outputs = tf.nn.dropout(outputs, dropout)

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [n_filters*len(filter_sizes), 1])
                b = tf.get_variable('b', [1])
                logits = tf.reshape(tf.matmul(outputs, W) + b, [-1])

        return logits

    def discriminator(self, x_real, x_fake, filter_sizes, n_filters, dropout, scope,
        wgan=False, eta=10):
        d_real = self.cnn(x_real, filter_sizes, n_filters, dropout, scope)
        d_fake = self.cnn(x_fake, filter_sizes, n_filters, dropout, scope, reuse=True)

        ones = tf.ones([tf.shape(d_real)[0]])
        zeros = tf.ones([tf.shape(d_fake)[0]])
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ones, logits=d_real)) + \
                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=zeros, logits=d_fake))

        ones = tf.ones([tf.shape(d_fake)[0]])
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=ones, logits=d_fake))
        return loss_d, loss_g


