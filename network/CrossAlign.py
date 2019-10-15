import tensorflow as tf

from network.nn import *
from network.BaseModel import BaseModel

class Model(BaseModel):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.build_placeholder()
        self.build_model(args)

    def build_model(self, args):
        # auto-encoder
        with tf.variable_scope('encoder_decoder'):
            # word embedding
            embedding = tf.get_variable('embedding', initializer=self.word_init)
            # embedding = tf.get_variable('embedding', [self.vocab_size, self.dim_emb])
            enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
            dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)
            with tf.variable_scope('projection'):
                # style information
                projection = {}
                projection['W'] = tf.get_variable('W', [self.dim_h, self.vocab_size])
                projection['b'] = tf.get_variable('b', [self.vocab_size])
            encoder = self.create_cell(self.dim_h, args.n_layers, self.dropout, 'encoder')
            decoder = self.create_cell(self.dim_h, args.n_layers, self.dropout, 'decoder')
            self.loss_rec, origin_info, transfer_info, real_sents = self.reconstruction(
                encoder, enc_inputs, self.labels,
                decoder, dec_inputs, self.targets, self.dec_mask, projection)
            fake_sents, _, self.rec_ids, self.tsf_ids = self.run_decoder(
                decoder, dec_inputs, embedding, projection, origin_info, transfer_info)

            # make the real sents and fake sents the same length
            if args.trim_padding:
                fake_sents = fake_sents[:, :1+self.batch_len, :]

        # discriminator
        with tf.variable_scope('discriminator'):
            self.loss_d1, loss_g1, self.loss_d0, loss_g0 = self.run_discriminator(
                real_sents, fake_sents, self.labels, args)

        #####   optimizer   #####
        self.loss_adv = loss_g0 + loss_g1
        self.loss = self.loss_rec + self.rho * self.loss_adv

        theta_eg = retrive_var(['encoder_decoder'])
        theta_d0 = retrive_var(['discriminator/negative'])
        theta_d1 = retrive_var(['discriminator/positive'])

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        grad, _ = zip(*opt.compute_gradients(self.loss, theta_eg))
        grad, _ = tf.clip_by_global_norm(grad, 30.0)

        self.optimize_tot = opt.apply_gradients(zip(grad, theta_eg))
        self.optimize_rec = opt.minimize(self.loss_rec, var_list=theta_eg)
        self.optimize_d0 = opt.minimize(self.loss_d0, var_list=theta_d0)
        self.optimize_d1 = opt.minimize(self.loss_d1, var_list=theta_d1)

        self.saver = tf.train.Saver()

    def reconstruction(self, encoder, enc_inputs, labels,
                       decoder, dec_inputs, targets, dec_mask, projection):
        labels = tf.reshape(labels, [-1, 1])
        batch_size = tf.shape(labels)[0]

        init_state = tf.concat([self.linear(labels, self.dim_y, scope='input_style'),
            tf.zeros([batch_size, self.dim_z])], 1)
        _, latent_vector = tf.nn.dynamic_rnn(encoder, enc_inputs, 
            initial_state=init_state, scope='encoder')


        latent_vector = latent_vector[:, self.dim_y:]
        origin_info = tf.concat([self.linear(labels, self.dim_y,
            scope='output_style'), latent_vector], 1)
        transfer_info = tf.concat([self.linear(1 - labels, self.dim_y,
            scope='output_style', reuse=True), latent_vector], 1)

        hiddens, _ = tf.nn.dynamic_rnn(decoder, dec_inputs,
            initial_state=origin_info, scope='decoder')

        real_sents = tf.concat([tf.expand_dims(origin_info, 1), hiddens], 1)

        hiddens = tf.nn.dropout(hiddens, self.dropout)
        hiddens = tf.reshape(hiddens, [-1, self.dim_h])
        logits = tf.matmul(hiddens, projection['W']) + projection['b']

        rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(targets, [-1]), logits=logits)
        rec_loss *= tf.reshape(dec_mask, [-1])
        rec_loss = tf.reduce_sum(rec_loss) / tf.to_float(batch_size)

        return rec_loss, origin_info, transfer_info, real_sents

    def run_discriminator(self, real_sents, fake_sents, labels, args):
        #####   discriminator   #####
        # a batch's first half consists of sentences of one style,
        # and second half of the other
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        batch_size = tf.shape(labels)[0]
        half = tf.to_int32(batch_size / 2)

        loss_d1, loss_g1 = self.discriminator(real_sents[:half], fake_sents[half:],
            filter_sizes, args.n_filters, self.dropout,
            scope='positive')

        loss_d0, loss_g0 = self.discriminator(real_sents[half:], fake_sents[:half],
            filter_sizes, args.n_filters, self.dropout,
            scope='negative')
        return loss_d1, loss_g1, loss_d0, loss_g0

    def _make_feed_dict(self, batch, mode='train'):
        feed_dict = {}
        if mode == 'train':
            dropout = self.dropout_rate
        else:
            dropout = 1.0

        feed_dict[self.dropout] = dropout
        feed_dict[self.batch_len] = batch.batch_len
        feed_dict[self.enc_inputs] = batch.enc_batch
        feed_dict[self.dec_inputs] = batch.dec_batch
        feed_dict[self.labels] = batch.labels

        feed_dict[self.targets] = batch.target_batch
        feed_dict[self.dec_mask] = batch.dec_padding_mask

        return feed_dict

    def run_train_step(self, sess, batch, accumulator, epoch = None):
        """Runs one training iteration. Returns a dictionary containing train op, 
           summaries, loss, global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_feed_dict(batch)

        to_return = {
            'loss_d0': self.loss_d0,
            'optimize_d0': self.optimize_d0,
            'loss_d1': self.loss_d1,
            'optimize_d1': self.optimize_d1,
        }
        results1 = sess.run(to_return, feed_dict)
        if results1['loss_d0'] < 1.2 and results1['loss_d1'] < 1.2:
            optimize = self.optimize_tot
        else:
            optimize = self.optimize_rec

        to_return = {
            'loss': self.loss,
            'loss_rec': self.loss_rec,
            'loss_adv': self.loss_adv,
            'optimize': optimize,
        }
        results2 = sess.run(to_return, feed_dict)
        results = {**results1, **results2}
        accumulator.add([results[name] for name in accumulator.names])

    def run_eval_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch, mode = 'eval')

        to_return = {
            'rec_ids': self.rec_ids,
            'tsf_ids': self.tsf_ids,
            'loss': self.loss,
            'loss_rec': self.loss_rec,
            'loss_adv': self.loss_adv,
            'loss_d1': self.loss_d1,
            'loss_d0': self.loss_d0,
        }
        return sess.run(to_return, feed_dict)

    def get_output_names(self):
        return ['loss', 'loss_rec', 'loss_adv', 'loss_d0', 'loss_d1']
