import os
import sys
import argparse

import glob
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

smoothie = SmoothingFunction().method1

import network
from utils import *
from vocab import Vocabulary
from config import load_arguments

folder_path = 'samples/imdb_amazon/finetune/*.txt'

args = load_arguments()
args.batch_size = 50
# sometimes you need switch the label if the gt order is switched
pos_label = 1
neg_label = 0
# pos_label = 0
# neg_label = 1

def calculate_bleu(transfer_, ref_):
    ref = []
    hypo = []
    for i in range(len(transfer_)):
        hypo.append(transfer_[i].split())
        ref.append([ref_[i].split()])
    bleu = corpus_bleu(ref, hypo)
    print("Bleu score on transfer sentences: %.4f" % bleu)

def make_batches(sents_):
    sents = [x.split() for x in sents_]
    lengths = [len(x) for x in sents]
    assert len(sents) % args.batch_size == 0

    begin = list(range(0, len(sents), args.batch_size))
    end = begin[1:] + [len(sents)]

    batches_text = []
    batches_len = []
    for i, j in zip(begin, end):
        batches_text.append(sents[i : j])
        batches_len.append(lengths[i : j])
    return batches_text, batches_len

def calculate_acc(sess, model, vocab, sents):

    total_correct = 0
    batches_text, batches_lens = make_batches(sents)
    for i in range(len(batches_text)):
        batch_ids = batch_text_to_ids(batches_text[i], vocab)

        # evaluate acc
        feed_dict = {model.input: batch_ids,
                     model.enc_lens: batches_lens[i],
                     model.dropout: 1.0}
        preds = sess.run(model.preds, feed_dict=feed_dict)

        if i < 10:
            total_correct += np.sum(preds == pos_label)
        else:
            total_correct += np.sum(preds == neg_label)
    print('The total acc is: %f' % (total_correct/1000))

def calculate_domain_acc(sess, model, vocab, sents):

    total_correct = 0
    batches_text, batches_lens = make_batches(sents)
    for i in range(len(batches_text)):
        batch_ids = batch_text_to_ids(batches_text[i], vocab)

        # evaluate acc
        feed_dict = {model.input: batch_ids,
                     model.enc_lens: batches_lens[i],
                     model.dropout: 1.0}
        preds = sess.run(model.preds, feed_dict=feed_dict)

        total_correct += np.sum(preds == 1)
    print('The total domain acc is: %f' % (total_correct/1000))


def calculate_gscore(sess, model, vocab, sents, refs):
    total_gscore = 0
    batches_text, batches_lens = make_batches(sents)
    for i in range(len(batches_text)):
        batch_ids = batch_text_to_ids(batches_text[i], vocab)

        # evaluate acc
        feed_dict = {model.input: batch_ids,
                     model.enc_lens: batches_lens[i],
                     model.dropout: 1.0}
        preds = sess.run(model.probs, feed_dict=feed_dict)

        if i < 10:
            confids = preds[:, pos_label]
        else:
            confids = preds[:, neg_label]
        for j in range(len(confids)):
            idx = i * 50 + j
            bleu = sentence_bleu([refs[idx].split()], sents[idx].split(), smoothing_function=smoothie)
            bleu *= 100
            confid = confids[j] * 100
            gscore = 2 * confid * bleu / (confid + bleu)
            total_gscore += gscore
    print('The total g score is: %f' % (total_gscore/1000))

def load_file(file_path):
    origin = []
    ref = []
    transfer = []
    f = open(file_path, 'r')
    lines = f.readlines()
    for line in lines:
        iterms = line.split('\t')
        origin.append(iterms[0].strip())
        ref.append(iterms[1].strip())
        transfer.append(iterms[2].strip())
    f.close()
    return origin, ref, transfer

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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with tf.variable_scope('target'):
        target_vocab = Vocabulary(args.target_vocab)
        target_classifier = eval('network.classifier.CNN_Model')(args, target_vocab, 'target')
        restore_classifier_by_path(target_classifier, args.target_classifier_path, 'target')

    # classifier for domain accuracy evaluation
    with tf.variable_scope('domain'):
        assert args.domain_adapt, "domain_adapt arg should be True."
        multi_vocab = Vocabulary(args.multi_vocab)
        domain_classifier = eval('network.classifier.CNN_Model')(args, multi_vocab, 'domain')
        restore_classifier_by_path(domain_classifier, args.domain_classifier_path, 'domain')

    files = glob.glob(folder_path)
    for file in files:
        origin, ref, transfer = load_file(file)
        print('##############################')
        print('Evaluating %s file.' % file)
        calculate_domain_acc(sess, domain_classifier, multi_vocab, transfer)
        calculate_acc(sess, target_classifier, target_vocab, transfer)
        calculate_bleu(transfer, ref)
        calculate_gscore(sess, target_classifier, target_vocab, transfer, ref)

