import os
import sys
import time
import random

import numpy as np
import tensorflow as tf

import network
from config import load_arguments
from vocab import *
from dataloader.multi_style_dataloader import MultiStyleDataloader

import ipdb

def create_model(sess, args, vocab):
    model = eval('network.classifier.CNN_Model')(args, vocab)
    if args.load_model:
        print('Loading model from', os.path.join(args.domain_classifier_path, 'model'))
        model.saver.restore(sess, os.path.join(args.domain_classifier_path, 'model'))
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    if not os.path.exists(args.domain_classifier_path):
            os.makedirs(args.domain_classifier_path)
    return model

def create_domain_classifier_batches(loader):
    new_loader = {}
    for split in ['train', 'valid', 'test']:
        source_batches = loader.get_batches(domain='source', mode=split)
        target_batches = loader.get_batches(domain='target', mode=split)

        batches = []
        for i in range(len(source_batches)):
            sbatch = source_batches[i]
            tbatch = target_batches[i%len(target_batches)]

            batch = type('', (), {})()
            # create labels # target.labels == 1, source.labels == 0
            batch.labels = np.zeros((len(sbatch.labels) + len(tbatch.labels)), dtype=np.int32)
            batch.labels[len(sbatch.labels):] = 1
            # create enc_lens
            batch.enc_lens = np.concatenate([sbatch.enc_lens, tbatch.enc_lens], axis=0)
            # create enc_batch
            batch.enc_batch = np.concatenate([sbatch.enc_batch, tbatch.enc_batch], axis=0)
            # create original_reviews
            batch.original_reviews = sbatch.original_reviews + tbatch.original_reviews

            batches.append(batch)
        new_loader[split] = batches
    return new_loader

if __name__ == '__main__':
    args = load_arguments()
    assert args.domain_adapt, "domain_adapt arg should be True."

    if not os.path.isfile(args.multi_vocab):
        build_unify_vocab([args.target_train_path, args.source_train_path], args.multi_vocab)
    multi_vocab = Vocabulary(args.multi_vocab)
    print('vocabulary size: %d' % multi_vocab.size)
    # load data
    loader = MultiStyleDataloader(args, multi_vocab)
    loader = create_domain_classifier_batches(loader)
    print("transfer dataset successfully!")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        model = create_model(sess, args, multi_vocab)

        batches = loader['train']

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

                    val_batches = loader['valid']
                    acc, _, _ = model.run_eval(sess, val_batches)
                    print('valid accuracy %.4f' % acc)
                    if acc > best_dev:
                        best_dev = acc
                        print('Saving model...')
                        model.saver.save(sess, os.path.join(args.domain_classifier_path, 'model'))

        test_batches = loader['test']
        acc, _, _ = model.run_eval(sess, test_batches)
        print('test accuracy %.4f' % acc)
