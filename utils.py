import os
import random
import logging

import tensorflow as tf
import numpy as np

from dataloader.style_dataloader import Example

logger = logging.getLogger(__name__)


class Accumulator(object):
    def __init__(self, div, names):
        self.div = div
        self.names = names
        self.n = len(self.names)
        self.values = [0] * self.n

    def clear(self):
        self.values = [0] * self.n

    def add(self, values):
        for i in range(self.n):
            self.values[i] += values[i] / self.div

    def output(self, prefix, write_dict, mode):
        if prefix:
            prefix += ' '
        for i in range(self.n):
            prefix += '%s %.2f' % (self.names[i], self.values[i])
            if i < self.n-1:
                prefix += ', '
        logger.info(prefix)

        add_summary_value(write_dict['writer'], self.names, self.values, write_dict['step'], mode)


def add_summary_value(writer, keys, values, iteration, mode, domain=''):
    if mode not in ['train', 'valid']:
        return 
        
    for key, value in zip(keys, values):
        key = os.path.join(mode, domain, key)
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, iteration)
    writer.flush()


def strip_eos(sents):
    new_ids, lengths = [], []
    for sent in sents:
        if '<eos>' in sent:
            sent = sent[:sent.index('<eos>')]
        new_ids.append(sent)
        lengths.append(len(sent))
    return new_ids, lengths


def write_output(origin, transfer, reconstruction, path, ref):
    t = open(path + '_transfer.txt', 'w')
    r = open(path + '_reconstruction.txt', 'w')
    for i in range(len(origin)):
        try:
            output = origin[i] + '\t' + ' '.join(ref[i][0]) + '\t' + ' '.join(transfer[i]) + '\n'
            t.write(output)
        except:
            pass
        try:
            output = origin[i] + '\t' + ' '.join(reconstruction[i]) + '\n'
            r.write(output)
        except:
            pass
    t.close()
    r.close()


def write_output_v0(origin, transfer, reconstruction, path):
    t = open(path + '_transfer.txt', 'w')
    r = open(path + '_reconstruction.txt', 'w')
    for i in range(len(origin)):
        try:
            output = origin[i] + '\t' + ' '.join(transfer[i]) + '\n'
            t.write(output)
        except:
            pass
        try:
            output = origin[i] + '\t' + ' '.join(reconstruction[i]) + '\n'
            r.write(output)
        except:
            pass
    t.close()
    r.close()


def batch_text_to_ids(batch_text, vocab):
    max_len = 5
    for x in batch_text:
        max_len = max(len(x), max_len)
    sent_ids = []
    for x in batch_text:
        sent_id = [vocab.word2id(word) for word in x]
        if len(sent_id) < max_len:
            sent_id += [vocab.word2id('<pad>')] * (max_len - len(sent_id))
        sent_ids.append(sent_id)
    return sent_ids

def batch_text_to_dec_inputs(batch_text, lengths, vocab):
    max_len = max(6, np.max(lengths) + 1)
    batch_size = len(batch_text)
    dec_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    target_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    dec_padding_mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for i, sent in enumerate(batch_text):
        sent_id = [vocab.word2id(word) for word in sent]
        inp = [vocab.word2id('<go>')] + sent_id
        target = sent_id + [vocab.word2id('<eos>')]
        if len(inp) < max_len:
            padding = [vocab.word2id('<pad>')] * (max_len - len(inp))
            inp.extend(padding)
            target.extend(padding)

        dec_batch[i, :] = inp
        target_batch[i, :] = target
        dec_padding_mask[i][:lengths[i]] = 1.0
    return dec_batch, target_batch, dec_padding_mask

def calculate_ppl(lm, sents):
    ppl = 0
    for sent in sents:
        score = lm.perplexity(sent)
        ppl += score
    ppl /= len(sents)
    return ppl