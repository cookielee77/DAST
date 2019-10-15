import os
import queue
import random
import codecs
import json
import glob
import numpy as np
import tensorflow as tf

class Example(object):
  """Class representing a train/val/test example for text summarization."""
  def __init__(self, review, label, vocab, hps):
    self.hps = hps
    review_words = []
    review_sentences = review

    review_words = review_sentences.split()
    if len(review_words) > hps.max_len: #:
        review_words = review_words[:hps.max_len]

    self.enc_input = [vocab.word2id(w) for w in review_words]  # list of word ids; OOVs are represented by the id for UNK token

    self.enc_len = len(review_words)  # store the length after truncation but before padding
    #self.enc_sen_len = [len(sentence_words) for sentence_words in review_words]
    self.label = int(label)
    self.original_reivew = review_sentences

  def pad_encoder_input(self, max_sen_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(self.enc_input) < max_sen_len:
            self.enc_input.append(pad_id)

class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.
    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id('<pad>') # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder



  def init_encoder_seq(self, example_list, hps):

    #print (example_list)

    #max_enc_seq_len = max(ex.enc_len for ex in example_list)
    for ex in example_list:
      ex.pad_encoder_input(hps.max_len, self.pad_id)
    
    batch_size = len(example_list)

    self.enc_batch = np.zeros((batch_size, hps.max_len), dtype=np.int32)
    self.enc_lens = np.zeros((batch_size), dtype=np.int32)
    self.labels = np.zeros((batch_size), dtype=np.int32)
    self.original_reviews = [ex.original_reivew for ex in example_list]

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.labels[i] = ex.label
      self.enc_batch[i,:] = np.array(ex.enc_input)[:]
      self.enc_lens[i] = ex.enc_len


class ClassificationBatcher(object):
    def __init__(self, hps, vocab):
        self._vocab = vocab
        self._hps = hps

        train_queue = self.fill_example_queue(hps.train_path, mode = 'train')
        valid_queue = self.fill_example_queue(hps.valid_path, mode = 'valid')
        test_queue = self.fill_example_queue(hps.test_path, mode = 'test')

        self.train_batch = self.create_batches(train_queue, mode="train")
        # update training checkpoint step
        checkpoint_step = int(len(self.train_batch) / (hps.train_checkpoint_frequency*50)) * 50
        hps.train_checkpoint_step = max(50, checkpoint_step)

        self.valid_batch = self.create_batches(valid_queue, mode="valid")
        self.test_batch = self.create_batches(test_queue, mode="test")

    def create_batches(self, queue, mode="train"):
        assert queue is not None
        all_batch = []

        if mode == 'train':
            random.shuffle(queue)

        begin = list(range(0, len(queue), self._hps.batch_size))
        end = begin[1:] + [len(queue)]

        for i, j in zip(begin, end):
            batch = queue[i : j]
            all_batch.append(Batch(batch, self._hps, self._vocab))
        return all_batch

    def get_batches(self, mode="train"):
        if mode == "train":
            random.shuffle(self.train_batch)
            return self.train_batch
        elif mode == 'valid':
            return self.valid_batch
        elif mode == 'test':
            return self.test_batch

    def fill_example_queue(self, data_path, mode = 'train'):

        new_queue =[]

        filelist = glob.glob(os.path.join(data_path, '*.txt'))  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty

        for f in filelist:
            reader = codecs.open(f, 'r', 'utf-8')
            while True:
                string_ = reader.readline()
                if not string_: break
                dict_example = json.loads(string_)
                review = dict_example["review"]
                if review.strip() =="":
                    continue
                score = dict_example["score"]

                if score not in [0, 1]:
                    raise ValueError('The score %d is not 0 or 1.' % score)
                
                example = Example(review, score, self._vocab, self._hps)
                new_queue.append(example)
        print('%s file has %d sentences.' % (mode, len(new_queue)))
        return new_queue
