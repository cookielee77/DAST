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
            self.loss_rec, origin_info, transfer_info = self.reconstruction(
                encoder, enc_inputs, self.labels,
                decoder, dec_inputs, self.targets, self.dec_mask, projection)
            _, soft_tsf_ids, self.rec_ids, self.tsf_ids = self.run_decoder(
                decoder, dec_inputs, embedding, projection, origin_info, transfer_info)

            # make the real sents and fake sents the same length
            if args.trim_padding:
                fake_probs = fake_probs[:, :1+self.batch_len, :]

        # discriminator
        with tf.variable_scope('discriminator'):
            classifier_embedding = tf.get_variable('embedding', initializer=self.word_init)
            # classifier_embedding = tf.get_variable('embedding', [self.vocab_size, self.dim_emb])
            # remove bos, use dec_inputs to avoid noises adding into enc_inputs
            real_sents = tf.nn.embedding_lookup(classifier_embedding, self.dec_inputs[:, 1:])
            fake_sents = tf.tensordot(soft_tsf_ids, classifier_embedding, [[2], [0]])
            fake_sents = fake_sents[:, :-1, :] # make the dimension the same as real sents

            # mask the sequences
            mask = tf.sequence_mask(self.enc_lens, self.max_len - 1, dtype = tf.float32)
            mask = tf.expand_dims(mask, -1)
            real_sents *= mask
            fake_sents *= mask

            self.loss_d, self.loss_g = self.run_discriminator(real_sents, fake_sents, self.labels, args)

        #####   optimizer   #####
        self.loss = self.loss_rec + self.rho * self.loss_g

        theta_eg = retrive_var(['encoder_decoder'])
        theta_d = retrive_var(['discriminator'])

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        grad, _ = zip(*opt.compute_gradients(self.loss, theta_eg))
        grad, _ = tf.clip_by_global_norm(grad, 30.0)

        self.optimize_tot = opt.apply_gradients(zip(grad, theta_eg))
        self.optimize_rec = opt.minimize(self.loss_rec, var_list=theta_eg)
        self.optimize_d = opt.minimize(self.loss_d, var_list=theta_d)

        self.saver = tf.train.Saver(max_to_keep=5)

    def reconstruction(self, encoder, enc_inputs, labels,
                       decoder, dec_inputs, targets, dec_mask, projection):
        labels = tf.reshape(labels, [-1, 1])

        _, latent_vector = tf.nn.dynamic_rnn(encoder, enc_inputs, 
            scope='encoder', dtype=tf.float32)

        latent_vector = latent_vector[:, self.dim_y:]
        origin_info = tf.concat([self.linear(labels, self.dim_y,
            scope='output_style'), latent_vector], 1)
        transfer_info = tf.concat([self.linear(1 - labels, self.dim_y,
            scope='output_style', reuse=True), latent_vector], 1)

        hiddens, _ = tf.nn.dynamic_rnn(decoder, dec_inputs,
            initial_state=origin_info, scope='decoder')

        hiddens = tf.nn.dropout(hiddens, self.dropout)
        hiddens = tf.reshape(hiddens, [-1, self.dim_h])
        logits = tf.matmul(hiddens, projection['W']) + projection['b']

        rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(targets, [-1]), logits=logits)
        rec_loss *= tf.reshape(dec_mask, [-1])

        # # ave over step #### still has some problems
        # rec_loss = tf.reshape(rec_loss, [batch_size, self.max_len])
        # rec_loss = tf.reduce_sum(rec_loss, axis = 1)
        # rec_loss = rec_loss / tf.to_float(self.enc_lens + 1)
        batch_size = tf.shape(labels)[0]
        rec_loss = tf.reduce_sum(rec_loss) / tf.to_float(batch_size)

        return rec_loss, origin_info, transfer_info

    def run_discriminator(self, real_sents, fake_sents, labels, args):
        #####   discriminator   #####
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        rev_labels = 1 - labels

        d_real = self.cnn(real_sents, filter_sizes, args.n_filters, self.dropout, 'classifer')
        d_fake = self.cnn(fake_sents, filter_sizes, args.n_filters, self.dropout, 'classifer', reuse=True)

        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=d_real))
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=rev_labels, logits=d_fake))

        return loss_d, loss_g

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
        feed_dict[self.enc_lens] = batch.enc_lens

        feed_dict[self.targets] = batch.target_batch
        feed_dict[self.dec_mask] = batch.dec_padding_mask

        return feed_dict

    def run_train_step(self, sess, batch, accumulator, epoch = None):
        """Runs one training iteration. Returns a dictionary containing train op, 
           summaries, loss, global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_feed_dict(batch)

        if epoch > self.pretrain_epochs:
            results1 = {'loss_d': 0.0}
        else:
            to_return = {
                'loss_d': self.loss_d,
                'optimize_d': self.optimize_d,
            }
            results1 = sess.run(to_return, feed_dict)

        if epoch > self.pretrain_epochs:
            optimize = self.optimize_tot
        else:
            optimize = self.optimize_rec

        to_return = {
            'loss': self.loss,
            'loss_rec': self.loss_rec,
            'loss_g': self.loss_g,
            'optimize': optimize,
        }
        results2 = sess.run(to_return, feed_dict)
        results = {**results1, **results2}
        accumulator.add([results[name] for name in accumulator.names])

    def run_eval_step(self, sess, batch, domain=None):
        feed_dict = self._make_feed_dict(batch, mode = 'eval')

        to_return = {
            'rec_ids': self.rec_ids,
            'tsf_ids': self.tsf_ids,
            'loss': self.loss,
            'loss_rec': self.loss_rec,
            'loss_g': self.loss_g,
            'loss_d': self.loss_d,
        }
        return sess.run(to_return, feed_dict)

    def get_output_names(self, domain=None):
        return ['loss', 'loss_rec', 'loss_g', 'loss_d']
