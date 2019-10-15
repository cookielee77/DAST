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
from vocab import Vocabulary, build_unify_vocab
from config import load_arguments
from dataloader.multi_style_dataloader import MultiStyleDataloader
from dataloader.online_dataloader import OnlineDataloader

smoothie = SmoothingFunction().method4

logger = logging.getLogger(__name__)

def evaluation(sess, args, batches, model, 
    classifier, classifier_vocab, domain_classifer, domain_vocab,
    output_path, write_dict, save_samples=False, mode='valid', domain=''):
    transfer_acc = 0
    domain_acc = 0
    origin_acc = 0
    total = 0
    domain_total =0
    ref = []
    ori_ref = []
    hypo = []
    origin = []
    transfer = []
    reconstruction = []
    accumulator = Accumulator(len(batches), model.get_output_names(domain))

    for batch in batches:
        results = model.run_eval_step(sess, batch, domain)
        accumulator.add([results[name] for name in accumulator.names])

        rec = [[domain_vocab.id2word(i) for i in sent] for sent in results['rec_ids']]
        rec, _ = strip_eos(rec)

        tsf = [[domain_vocab.id2word(i) for i in sent] for sent in results['tsf_ids']]
        tsf, lengths = strip_eos(tsf)

        reconstruction.extend(rec)
        transfer.extend(tsf)
        hypo.extend(tsf)
        origin.extend(batch.original_reviews)
        for x in batch.original_reviews:
            ori_ref.append([x.split()])
        for x in batch.references:
            ref.append([x.split()])

        # tansfer the output sents into classifer ids for evaluation
        tsf_ids = batch_text_to_ids(tsf, classifier_vocab)
        # evaluate acc
        feed_dict = {classifier.input: tsf_ids,
                     classifier.enc_lens: lengths,
                     classifier.dropout: 1.0}
        preds = sess.run(classifier.preds, feed_dict=feed_dict)
        trans_label = batch.labels == 0
        transfer_acc += np.sum(trans_label == preds)
        total += len(trans_label)

        # evaluate domain acc
        if domain == 'target':
            domian_ids = batch_text_to_ids(tsf, domain_vocab)
            feed_dict = {domain_classifier.input: domian_ids,
                         domain_classifier.enc_lens: lengths,
                         domain_classifier.dropout: 1.0}
            preds = sess.run(domain_classifier.preds, feed_dict=feed_dict)
            domain_acc += np.sum(preds == 1)
            domain_total += len(preds)

    accumulator.output(mode, write_dict, mode)
    if domain == 'target':
        output_domain_acc = (domain_acc / float(domain_total))
        logger.info("domain acc: %.4f" % output_domain_acc)
    output_acc = (transfer_acc / float(total))
    logger.info("transfer acc: %.4f" % output_acc)
    bleu = corpus_bleu(ref, hypo, smoothing_function=smoothie)
    logger.info("Bleu score: %.4f" % bleu)

    add_summary_value(write_dict['writer'], ['acc', 'bleu'], [output_acc, bleu], write_dict['step'], mode, domain)

    if mode == 'online-test':
        bleu = corpus_bleu(ori_ref, hypo, smoothing_function=smoothie)
        logger.info("Bleu score on original sentences: %.4f" % bleu)
        if save_samples:
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

# elimiate the first variable scope, and restore the classifier from the path
def restore_classifier_by_path(classifier, classifier_path, scope):
    new_vars = {}
    for var in classifier.params:
        pos = var.name.find('/')
        # eliminate the first variable scope, e.g., target, source
        new_vars[var.name[pos+1:-2]] = var
    saver = tf.train.Saver(new_vars)
    saver.restore(sess, os.path.join(classifier_path, 'model'))
    logger.info("-----%s classifier model loading from %s successfully!-----" % (scope, classifier_path))

if __name__ == '__main__':
    args = load_arguments()
    assert args.domain_adapt, "domain_adapt arg should be True."

    if not os.path.isfile(args.multi_vocab):
        build_unify_vocab([args.target_train_path, args.source_train_path], args.multi_vocab)
    multi_vocab = Vocabulary(args.multi_vocab)
    logger.info('vocabulary size: %d' % multi_vocab.size)

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
    loader = MultiStyleDataloader(args, multi_vocab)
    # create a folder for data samples
    source_output_path = os.path.join(args.logDir, 'domain_adapt', 'source')
    if not os.path.exists(source_output_path):
        os.makedirs(source_output_path)
    target_output_path = os.path.join(args.logDir, 'domain_adapt', 'target')
    if not os.path.exists(target_output_path):
        os.makedirs(target_output_path)

    # whether use online dataset for testing
    if args.online_test:
        online_data = OnlineDataloader(args, multi_vocab)
        online_data = online_data.online_test
        output_online_path = os.path.join(args.logDir, 'domain_adapt', 'online-test')
        if not os.path.exists(output_online_path):
            os.mkdir(output_online_path)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # create style transfer model
        model = create_model(sess, args, multi_vocab)

        # vocabulary for classifer evalution
        with tf.variable_scope('target'):
            target_vocab = Vocabulary(args.target_vocab)
            target_classifier = eval('network.classifier.CNN_Model')(args, target_vocab, 'target')
            restore_classifier_by_path(target_classifier, args.target_classifier_path, 'target')

        with tf.variable_scope('domain'):
            domain_classifier = eval('network.classifier.CNN_Model')(args, multi_vocab, 'domain')
            restore_classifier_by_path(domain_classifier, args.domain_classifier_path, 'domain')

        # load training dataset
        source_batches = loader.get_batches(domain='source', mode='train')
        target_batches = loader.get_batches(domain='target', mode='train')

        start_time = time.time()
        step = 0
        accumulator = Accumulator(args.train_checkpoint_step, model.get_output_names('all'))
        learning_rate = args.learning_rate

        best_bleu = 0.0
        acc_cut = 0.90
        gamma = args.gamma_init
        for epoch in range(1, 1+args.max_epochs):
            logger.info('--------------------epoch %d--------------------' % epoch)
            logger.info('learning_rate: %.4f  gamma: %.4f' % (learning_rate, gamma))

            # multi dataset training
            source_len = len(source_batches)
            target_len = len(target_batches)
            iter_len = max(source_len, target_len)
            for i in range(iter_len):
                model.run_train_step(sess, 
                    target_batches[i % target_len], source_batches[i % source_len], accumulator, epoch)

                step += 1
                write_dict['step'] = step
                if step % args.train_checkpoint_step == 0:
                    accumulator.output('step %d, time %.0fs,'
                        % (step, time.time() - start_time), write_dict, 'train')
                    accumulator.clear()

                    # validation
                    val_batches = loader.get_batches(domain='target', mode='valid')
                    logger.info('---evaluating target domain:')
                    acc, bleu = evaluation(sess, args, val_batches, model,
                        target_classifier, target_vocab, domain_classifier, multi_vocab,
                        os.path.join(target_output_path, 'epoch%d' % epoch), write_dict,
                        mode='valid', domain='target')

                    # evaluate online test dataset
                    if args.online_test and acc > acc_cut and bleu > best_bleu:
                        best_bleu = bleu
                        save_samples = epoch > args.pretrain_epochs
                        online_acc, online_bleu = evaluation(sess, args, online_data, model,
                            target_classifier, target_vocab, domain_classifier, multi_vocab,
                            os.path.join(output_online_path, 'step%d' % step), write_dict,
                            mode='online-test', domain='target', save_samples=save_samples)

                    if args.save_model:
                        logger.info('Saving style transfer model...')
                        model.saver.save(sess, os.path.join(args.styler_path, 'model'))

        # testing
        test_batches = loader.get_batches(domain='target', mode='test')
        logger.info('---testing target domain:')
        evaluation(sess, args, test_batches, model, 
            target_classifier, target_vocab, domain_classifier, multi_vocab,
            os.path.join(target_output_path, 'test'), write_dict, mode='test', domain='target')
