import tensorflow as tf

from network.nn import *
from network.ControlGen import Model as BaseModel

class Model(BaseModel):
    def __init__(self, args, vocab):
        self.dim_d = args.dim_d
        self.alpha = args.alpha
        super().__init__(args, vocab)

    def build_placeholder(self):
        self.dropout = tf.placeholder(tf.float32,
            name='dropout')

        # target_data
        self.td_batch_len = tf.placeholder(tf.int32,
            name='td_batch_len')
        self.td_enc_inputs = tf.placeholder(tf.int32, [None, None], #size * len
            name='td_enc_inputs')
        self.td_dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='td_dec_inputs')
        self.td_targets = tf.placeholder(tf.int32, [None, None],
            name='td_targets')
        self.td_dec_mask = tf.placeholder(tf.float32, [None, None],
            name='td_dec_mask')
        self.td_labels = tf.placeholder(tf.float32, [None],
            name='td_labels')
        self.td_enc_lens = tf.placeholder(tf.float32, [None],
            name='td_enc_lens')

        # source_data
        self.sd_batch_len = tf.placeholder(tf.int32,
            name='sd_batch_len')
        self.sd_enc_inputs = tf.placeholder(tf.int32, [None, None], #size * len
            name='sd_enc_inputs')
        self.sd_dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='sd_dec_inputs')
        self.sd_targets = tf.placeholder(tf.int32, [None, None],
            name='sd_targets')
        self.sd_dec_mask = tf.placeholder(tf.float32, [None, None],
            name='sd_dec_mask')
        self.sd_labels = tf.placeholder(tf.float32, [None],
            name='sd_labels')
        self.sd_enc_lens = tf.placeholder(tf.float32, [None],
            name='sd_enc_lens')


    def build_model(self, args):
        with tf.variable_scope('encoder_decoder'):
            target_dv = self.linear(tf.ones([tf.shape(self.td_labels)[0], 1]), self.dim_d, scope='domain_vector')
            source_dv = self.linear(tf.zeros([tf.shape(self.sd_labels)[0], 1]), self.dim_d, scope='domain_vector', reuse=True)
            # domain vector loss
            pos_vector = self.linear(tf.ones([1,1]), self.dim_d, scope='domain_vector', reuse=True)
            neg_vector = self.linear(tf.zeros([1,1]), self.dim_d, scope='domain_vector', reuse=True)
            self.domain_loss = tf.nn.l2_loss(pos_vector - neg_vector)

        outputs = self.style_transfer_model(args, self.td_enc_inputs, target_dv, self.td_dec_inputs,
            self.td_targets, self.td_dec_mask, self.td_labels, self.td_enc_lens,
            scope = 'target')
        self.td_loss_rec, self.td_loss_d, self.td_loss_g, self.td_tsf_ids, self.td_rec_ids = outputs

        outputs = self.style_transfer_model(args, self.sd_enc_inputs, source_dv, self.sd_dec_inputs,
            self.sd_targets, self.sd_dec_mask, self.sd_labels, self.sd_enc_lens,
            scope = 'source')
        self.sd_loss_rec, self.sd_loss_d, self.sd_loss_g, self.sd_tsf_ids, self.sd_rec_ids = outputs

        # optimizer
        self.loss_rec = self.td_loss_rec + self.sd_loss_rec + self.alpha * self.domain_loss
        self.loss_g = self.td_loss_g + self.sd_loss_g
        self.loss_d = self.td_loss_d + self.sd_loss_d

        self.loss = self.loss_rec + self.rho * self.loss_g

        theta_eg = retrive_var(['encoder_decoder'])
        theta_d = retrive_var(['discriminator'])

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        grad, _ = zip(*opt.compute_gradients(self.loss, theta_eg))
        grad, _ = tf.clip_by_global_norm(grad, 30.0)

        self.optimize_tot = opt.apply_gradients(zip(grad, theta_eg))
        self.optimize_rec = opt.minimize(self.loss_rec, var_list=theta_eg)
        self.optimize_d = opt.minimize(self.loss_d, var_list=theta_d)


    def style_transfer_model(self, args, enc_input_ids, domain_vector,
        dec_input_ids, targets, dec_mask, labels, enc_lens, scope = None):
        # auto-encoder
        with tf.variable_scope('encoder_decoder', reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable('embedding', initializer=self.word_init)
            enc_inputs = tf.nn.embedding_lookup(embedding, enc_input_ids)
            dec_inputs = tf.nn.embedding_lookup(embedding, dec_input_ids)

            with tf.variable_scope('projection'):
                # style information
                projection = {}
                projection['W'] = tf.get_variable('W', [self.dim_h, self.vocab_size])
                projection['b'] = tf.get_variable('b', [self.vocab_size])
            encoder = self.create_cell(self.dim_h, args.n_layers, self.dropout, 'encoder')
            decoder = self.create_cell(self.dim_h, args.n_layers, self.dropout, 'decoder')

            loss_rec, origin_info, transfer_info = self.reconstruction(
                encoder, enc_inputs, labels, domain_vector,
                decoder, dec_inputs, targets, dec_mask, projection)
            _, soft_tsf_ids, rec_ids, tsf_ids = self.run_decoder(
                decoder, dec_inputs, embedding, projection, origin_info, transfer_info)

        # discriminator
        with tf.variable_scope("discriminator"):
            with tf.variable_scope(scope):
                classifier_embedding = tf.get_variable('embedding', initializer=self.word_init)
                # remove bos, use dec_inputs to avoid noises adding into enc_inputs
                real_sents = tf.nn.embedding_lookup(classifier_embedding, dec_input_ids[:, 1:])
                fake_sents = tf.tensordot(soft_tsf_ids, classifier_embedding, [[2], [0]])
                fake_sents = fake_sents[:, :-1, :] # make the dimension the same as real sents

                # mask the sequences
                mask = tf.sequence_mask(enc_lens, self.max_len - 1, dtype = tf.float32)
                mask = tf.expand_dims(mask, -1)
                real_sents *= mask
                fake_sents *= mask

                loss_d, loss_g = self.run_discriminator(real_sents, fake_sents, labels, args)

        return [loss_rec, loss_d, loss_g, tsf_ids, rec_ids]

    def reconstruction(self, encoder, enc_inputs, labels, domain_vector,
                       decoder, dec_inputs, targets, dec_mask, projection):
        labels = tf.reshape(labels, [-1, 1])

        _, latent_vector = tf.nn.dynamic_rnn(encoder, enc_inputs, 
            scope='encoder', dtype=tf.float32)

        # concate style, latent, domain vectors
        latent_vector = latent_vector[:, self.dim_y + self.dim_d:]
        origin_info = tf.concat([self.linear(labels, self.dim_y,
            scope='output_style'), domain_vector, latent_vector], 1)
        transfer_info = tf.concat([self.linear(1 - labels, self.dim_y,
            scope='output_style', reuse=True), domain_vector, latent_vector], 1)

        hiddens, _ = tf.nn.dynamic_rnn(decoder, dec_inputs,
            initial_state=origin_info, scope='decoder')

        hiddens = tf.nn.dropout(hiddens, self.dropout)
        hiddens = tf.reshape(hiddens, [-1, self.dim_h])
        logits = tf.matmul(hiddens, projection['W']) + projection['b']

        rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(targets, [-1]), logits=logits)
        rec_loss *= tf.reshape(dec_mask, [-1])

        batch_size = tf.shape(labels)[0]
        rec_loss = tf.reduce_sum(rec_loss) / tf.to_float(batch_size)

        return rec_loss, origin_info, transfer_info

    def _make_feed_dict(self, td_batch, sd_batch, mode='train'):
        feed_dict = {}
        if mode == 'train':
            dropout = self.dropout_rate
        else:
            dropout = 1.0

        feed_dict[self.dropout] = dropout

        # target data
        if td_batch is not None:
            feed_dict[self.td_batch_len] = td_batch.batch_len
            feed_dict[self.td_enc_inputs] = td_batch.enc_batch
            feed_dict[self.td_dec_inputs] = td_batch.dec_batch
            feed_dict[self.td_labels] = td_batch.labels
            feed_dict[self.td_enc_lens] = td_batch.enc_lens
            feed_dict[self.td_targets] = td_batch.target_batch
            feed_dict[self.td_dec_mask] = td_batch.dec_padding_mask

        # source data
        if sd_batch is not None:
            feed_dict[self.sd_batch_len] = sd_batch.batch_len
            feed_dict[self.sd_enc_inputs] = sd_batch.enc_batch
            feed_dict[self.sd_dec_inputs] = sd_batch.dec_batch
            feed_dict[self.sd_labels] = sd_batch.labels
            feed_dict[self.sd_enc_lens] = sd_batch.enc_lens
            feed_dict[self.sd_targets] = sd_batch.target_batch
            feed_dict[self.sd_dec_mask] = sd_batch.dec_padding_mask

        return feed_dict

    def run_train_step(self, sess, td_batch, sd_batch, accumulator, epoch = None):
        """Runs one training iteration. Returns a dictionary containing train op, 
           summaries, loss, global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_feed_dict(td_batch, sd_batch)

        if epoch > self.pretrain_epochs:
            results1 = {'td_loss_d': 0.0, 'sd_loss_d': 0.0}
        else:
            to_return = {
                'td_loss_d': self.td_loss_d,
                'sd_loss_d': self.sd_loss_d,
                'optimize_d': self.optimize_d,
            }
            results1 = sess.run(to_return, feed_dict)

        if epoch > self.pretrain_epochs:
            optimize = self.optimize_tot
        else:
            optimize = self.optimize_rec

        to_return = {
            'sd_loss_rec': self.sd_loss_rec,
            'sd_loss_g': self.sd_loss_g,
            'td_loss_rec': self.td_loss_rec,
            'td_loss_g': self.td_loss_g,
            'domain_loss': self.domain_loss,
            'optimize': optimize,
        }
        results2 = sess.run(to_return, feed_dict)
        results = {**results1, **results2}
        accumulator.add([results[name] for name in accumulator.names])

    def run_eval_step(self, sess, batch, domain=None):
        if domain == 'source':
            feed_dict = self._make_feed_dict(None, batch, mode = 'eval')

            to_return = {
                'rec_ids': self.sd_rec_ids,
                'tsf_ids': self.sd_tsf_ids,
                'sd_loss_rec': self.sd_loss_rec,
                'sd_loss_g': self.sd_loss_g,
                'sd_loss_d': self.sd_loss_d,
            }
        elif domain == 'target':
            feed_dict = self._make_feed_dict(batch, None, mode = 'eval')

            to_return = {
                'rec_ids': self.td_rec_ids,
                'tsf_ids': self.td_tsf_ids,
                'td_loss_rec': self.td_loss_rec,
                'td_loss_g': self.td_loss_g,
                'td_loss_d': self.td_loss_d,
            }
        else:
            raise ValueError('Wrong domain name: %s.' % domain)
        return sess.run(to_return, feed_dict)

    def get_output_names(self, domain=None):
        if domain == 'source':
            return ['sd_loss_rec', 'sd_loss_g', 'sd_loss_d']
        elif domain == 'target':
            return ['td_loss_rec', 'td_loss_g', 'td_loss_d']
        elif domain == 'all':
            return ['td_loss_rec', 'td_loss_g', 'td_loss_d',
                    'sd_loss_rec', 'sd_loss_g', 'sd_loss_d',
                    'domain_loss']
        else:
            raise ValueError('Wrong domain name: %s.' % domain)
