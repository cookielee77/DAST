import os
import tensorflow as tf

class CNN_Model(object):

    def __init__(self, args, vocab, scope=''):
        # scope is the parent scope outside of the CNN
        with tf.variable_scope("classifer"):
            dim_emb = args.dim_emb
            filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
            n_filters = args.n_filters

            self.learning_rate = args.learning_rate
            self.dropout = tf.placeholder(tf.float32,
                name='dropout')
            self.input = tf.placeholder(tf.int32, [None, None],    #batch_size * max_len
                name='input')
            self.enc_lens = tf.placeholder(tf.int32, [None],
                name='enc_lens')
            self.label = tf.placeholder(tf.int32, [None],
                name='label')
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
            x = tf.nn.embedding_lookup(embedding, self.input)

            batch_len = tf.shape(x)[1]
            mask = tf.expand_dims(tf.sequence_mask(self.enc_lens, batch_len, tf.float32), -1)
            x *= mask
            
            self.logits = self.cnn(x, filter_sizes, n_filters, self.dropout, 'cnn')

            self.probs = tf.nn.softmax(self.logits, -1)
            self.preds = tf.argmax(self.probs, -1, output_type=tf.int32)
            self.correct_preds = tf.equal(self.preds, self.label)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, self.global_step)
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=os.path.join(scope, 'classifer'))
        self.saver = tf.train.Saver(self.params)

    def leaky_relu(self, x, alpha=0.01):
        return tf.maximum(alpha * x, x)

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
                    h = self.leaky_relu(conv + b)
                    # max pooling over time
                    pooled = tf.reduce_max(h, reduction_indices=1)
                    pooled = tf.reshape(pooled, [-1, n_filters])
                    outputs.append(pooled)
            outputs = tf.concat(outputs, 1)
            outputs = tf.nn.dropout(outputs, dropout)

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [n_filters*len(filter_sizes), 2])
                b = tf.get_variable('b', [2])
                logits = tf.matmul(outputs, W) + b

        return logits

    def _make_train_feed_dict(self, batch):
        feed_dict = {}
        feed_dict[self.input] = batch.enc_batch
        feed_dict[self.label] = batch.labels
        feed_dict[self.enc_lens] = batch.enc_lens
        feed_dict[self.dropout] = 0.5
        return feed_dict

    def _make_test_feed_dict(self, batch):
        feed_dict = {}
        feed_dict[self.input] = batch.enc_batch
        feed_dict[self.label] = batch.labels
        feed_dict[self.enc_lens] = batch.enc_lens
        feed_dict[self.dropout] = 1.0
        return feed_dict

    def run_train_step(self, sess, batch):
        feed_dict = self._make_train_feed_dict(batch)
        to_return = {
            'train_op': self.train_op,
            'loss': self.loss,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def run_eval(self, sess, batches):
        """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
        correct = 0
        total = 0
        error_list =[]
        error_label = []
        for batch in batches:
            feed_dict = self._make_test_feed_dict(batch)
            to_return = {
                'predictions': self.correct_preds,
                'pred_confs': self.probs,
                'preds': self.preds
            }
            results = sess.run(to_return, feed_dict)

            for i in range(len(results['predictions'])):
                total += 1
                if results['predictions'][i]:
                    correct +=1
                else:
                    error_label.append(results['predictions'][i])
                    error_list.append(batch.original_reviews[i])

        return correct/total, error_list, error_label

    def run_eval_conf(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_test_feed_dict(batch)
        to_return = {
            'pred_conf': self.probs,
        }
        return sess.run(to_return, feed_dict)
