import os
import sys
import time
import random
import logging

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

import network
from utils import *
from vocab import Vocabulary, build_vocab
from config import load_arguments
from dataloader.style_dataloader import StyleDataloader
from dataloader.online_dataloader import OnlineDataloader

smoothie = SmoothingFunction().method4

logger = logging.getLogger(__name__)

def evaluation(sess, args, vocab, batches, model, classifier, output_path, write_dict, mode = 'valid'):
    transfer_acc = 0
    origin_acc = 0
    total = 0
    ref = []
    ori_ref = []
    hypo = []
    origin = []
    transfer = []
    reconstruction = []
    accumulator = Accumulator(len(batches), model.get_output_names())

    for batch in batches:
        results = model.run_eval_step(sess, batch)
        accumulator.add([results[name] for name in accumulator.names])

        rec = [[vocab.id2word(i) for i in sent] for sent in results['rec_ids']]
        rec, _ = strip_eos(rec)

        tsf = [[vocab.id2word(i) for i in sent] for sent in results['tsf_ids']]
        tsf, lengths = strip_eos(tsf)

        reconstruction.extend(rec)
        transfer.extend(tsf)
        hypo.extend(tsf)
        origin.extend(batch.original_reviews)
        for x in batch.original_reviews:
            ori_ref.append([x.split()])
        for x in batch.references:
            ref.append([x.split()])

        # evaluate acc
        feed_dict = {classifier.input: results['tsf_ids'],
                     classifier.enc_lens: lengths,
                     classifier.dropout: 1.0}
        preds = sess.run(classifier.preds, feed_dict=feed_dict)
        trans_label = batch.labels == 0
        transfer_acc += np.sum(trans_label == preds)
        total += len(trans_label)

    accumulator.output(mode, write_dict, mode)
    output_acc = (transfer_acc / float(total))
    logger.info("transfer acc: %.4f" % output_acc)
    bleu = corpus_bleu(ref, hypo, smoothing_function=smoothie)
    logger.info("Bleu score: %.4f" % bleu)

    add_summary_value(write_dict['writer'], ['acc', 'bleu'], [output_acc, bleu], write_dict['step'], mode)

    if mode == 'online-test':
        bleu = corpus_bleu(ori_ref, hypo, smoothing_function=smoothie)
        logger.info("Bleu score on original sentences: %.4f" % bleu)
        write_output(origin, transfer, reconstruction, output_path, ref)
    elif args.save_samples:
        write_output_v0(origin, transfer, reconstruction, output_path)

    return output_acc, bleu


def create_model(sess, args, vocab):
    model = eval('network.' + args.network + '.Model')(args, vocab)
    if args.load_model:
        logger.info('-----Loading styler model from: %s.-----' % os.path.join(args.styler_path, 'model'))
        model.saver.restore(sess, os.path.join(args.styler_path, 'model'))
    else:
        logger.info('-----Creating styler model with fresh parameters.-----')
        sess.run(tf.global_variables_initializer())
    if not os.path.exists(args.styler_path):
            os.makedirs(args.styler_path)
    return model

if __name__ == '__main__':
    args = load_arguments()

    if not os.path.isfile(args.vocab):
        build_vocab(args.train_path, args.vocab)
    vocab = Vocabulary(args.vocab)
    logger.info('vocabulary size: %d' % vocab.size)

    # use tensorboard
    if args.suffix:
        tensorboard_dir = os.path.join(args.logDir, 'tensorboard', args.suffix)
    else:
        tensorboard_dir = os.path.join(args.logDir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    write_dict = {
    'writer': tf.summary.FileWriter(logdir=tensorboard_dir, filename_suffix=args.suffix),
    'step': 0
    }

    # load data
    loader = StyleDataloader(args, vocab)
    # create a folder for data samples
    output_path = os.path.join(args.logDir, args.dataset)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # whether use online test data
    if args.online_test:
        online_data = OnlineDataloader(args, vocab)
        online_data = online_data.online_test
        output_online_path = os.path.join(args.logDir, args.dataset, 'online-test')
        if not os.path.exists(output_online_path):
            os.mkdir(output_online_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # create style transfer model
        model = create_model(sess, args, vocab)
        classifier = eval('network.classifier.CNN_Model')(args, vocab)
        # load pretrained classifer for test
        classifier.saver.restore(sess, os.path.join(args.classifier_path, 'model'))
        logger.info("-----%s classifier model loading from %s successfully!-----" % (args.dataset, args.classifier_path))

        batches = loader.get_batches(mode='train')
        start_time = time.time()
        step = 0
        accumulator = Accumulator(args.train_checkpoint_step, model.get_output_names())
        learning_rate = args.learning_rate

        best_bleu = 0.0
        acc_cut = 0.90
        gamma = args.gamma_init
        for epoch in range(1, 1+args.max_epochs):
            logger.info('--------------------epoch %d--------------------' % epoch)
            logger.info('learning_rate: %.4f  gamma: %.4f' % (learning_rate, gamma))

            # this is to balance the training iteration number
            # 2835 is the iteration number when full dataset is used
            total_batch = max(2835, len(batches))
            for i in range(total_batch):
                model.run_train_step(sess, batches[i%len(batches)], accumulator, epoch)

                step += 1
                write_dict['step'] = step
                if step % 1000 == 0:
                # if step % args.train_checkpoint_step == 0:
                    accumulator.output('step %d, time %.0fs,'
                        % (step, time.time() - start_time), write_dict, 'train')
                    accumulator.clear()

                    # validation
                    val_batches = loader.get_batches(mode='valid')
                    acc, bleu = evaluation(sess, args, vocab, val_batches, model, classifier,
                        os.path.join(output_path, 'epoch%d' % epoch), write_dict, mode='valid')

                    # evaluate online test dataset
                    if args.online_test and acc > acc_cut and bleu > best_bleu:
                        best_bleu = bleu
                        acc, bleu = evaluation(sess, args, vocab, online_data, model, classifier,
                            os.path.join(output_online_path, 'step%d' % step), write_dict, mode='online-test')

                    if args.save_model:
                        logger.info('Saving style transfer model...')
                        model.saver.save(sess, os.path.join(args.styler_path, 'model'))

        # testing
        test_batches = loader.get_batches(mode='test')
        evaluation(sess, args, vocab, test_batches, model, classifier,
            os.path.join(output_path, 'test'), write_dict, mode='test')
