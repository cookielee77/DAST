import os
import string
import glob
import codecs

from nltk.tokenize import word_tokenize

from dataloader.style_dataloader import StyleDataloader, Example, Batch

class OnlineDataloader(StyleDataloader):
    def __init__(self, hps, vocab):
        self._vocab = vocab
        self._hps = hps

        data_path = os.path.join(hps.dataDir, hps.dataset, 'online-test')

        assert os.path.exists(data_path), "No online %s test dataset in %s" % (hps.dataset, data_path)

        # target dataset
        print('loading %s online test dataset: %s' % (hps.dataset, data_path))
        queue = self.fill_example_queue(data_path)
        self.online_test = self.create_batch(queue)

    def fill_example_queue(self, data_path):
        queue = []
        filelist = glob.glob(os.path.join(data_path, 'reference.*'))  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty

        for f in filelist:
            reader = codecs.open(f, 'r', 'utf-8')
            score = int(f[-1])
            while True:
                string_ = reader.readline()
                if not string_: break
                review, tsf = string_.split('\t')
                # processing the data, fix punctuation problem and tokenization problem in the annotated data
                review = review.strip()
                tsf = tsf.strip()
                if review[-1] != tsf[-1] and review[-1] in string.punctuation:
                    tsf += review[-1]
                review = word_tokenize(review)
                tsf = word_tokenize(tsf)

                example = Example(' '.join(review), ' '.join(tsf), score, self._vocab, self._hps)
                queue.append(example)

        print('Online file has %d total unique sentences.' % len(queue))
        return queue

    def create_batch(self, queue):
        all_batch = []
        batch_size = 100

        begin = list(range(0, len(queue), batch_size))
        end = begin[1:] + [len(queue)]

        for i, j in zip(begin, end):
            batch = queue[i : j]
            all_batch.append(Batch(batch, self._hps, self._vocab))

        # assert len(all_batch) * batch_size == len(queue)

        return all_batch
