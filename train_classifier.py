import os
import sys
import time
import random

import numpy as np
import tensorflow as tf

import network
from config import load_arguments
from vocab import Vocabulary, build_vocab
from dataloader.cnn_dataloader import ClassificationBatcher

def create_model(sess, args, vocab):
    model = eval('network.classifier.CNN_Model')(args, vocab)
    if args.load_model:
        print('Loading model from', os.path.join(args.classifier_path, 'model'))
        model.saver.restore(sess, os.path.join(args.classifier_path, 'model'))
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    if not os.path.exists(args.classifier_path):
            os.makedirs(args.classifier_path)
    return model

if __name__ == '__main__':
    args = load_arguments()

    if not os.path.isfile(args.vocab):
        build_vocab(args.train_path, args.vocab)
    vocab = Vocabulary(args.vocab)
    print('vocabulary size', vocab.size)

    loader = ClassificationBatcher(args, vocab)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        model = create_model(sess, args, vocab)

        batches = loader.get_batches(mode='train')

        start_time = time.time()
        loss = 0.0
        best_dev = float('-inf')
        learning_rate = args.learning_rate

        for epoch in range(1, 1+args.max_epochs):
            print('--------------------epoch %d--------------------' % epoch)

            for batch in batches:
                results = model.run_train_step(sess, batch)
                step_loss = results['loss']
                loss += step_loss / args.train_checkpoint_step

                if results['global_step'] % args.train_checkpoint_step == 0:
                    print('iteration %d, time %.0fs, loss %.4f' \
                        % (results['global_step'], time.time() - start_time, loss))
                    loss = 0.0

                    val_batches = loader.get_batches(mode='valid')
                    acc, _, _ = model.run_eval(sess, val_batches)
                    print('valid accuracy %.4f' % acc)
                    if acc > best_dev:
                        best_dev = acc
                        print('Saving model...')
                        model.saver.save(sess, os.path.join(args.classifier_path, 'model'))

        test_batches = loader.get_batches(mode='test')
        acc, _, _ = model.run_eval(sess, test_batches)
        print('test accuracy %.4f' % acc)
