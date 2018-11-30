import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import sys

from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from sklearn.datasets import fetch_20newsgroups
import unicodedata
import gensim
import string
import re
import collections
import logging
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def replaceAll(s, d):
    for k, v in d:
        s = s.replace(k, v)

    return s

class character_rnn(object):
    '''
    sample character-level RNN by Shang Gao

    parameters:
      - seq_len: integer (default: 200)
        number of characters in input sequence
      - first_read: integer (default: 50)
        number of characters to first read before attempting to predict next character
      - rnn_size: integer (default: 200)
        number of rnn cells

    methods:
      - train(text,iterations=100000)
        train network on given text
    '''

    def __init__(self, corpusfp, seq_len=200, first_read=50, rnn_size=200):

        self.seq_len = seq_len
        self.first_read = first_read

        self.corpusfp = corpusfp
        self.dataset = None
        self.sentences = None

        self.puncToTag = [('.', ' <STOP>.'), ('?', ' <QUEST>.'), ('!', ' <BANG>.')]
        self.tagToPunc = [('<STOP>', '.'), ('<QUEST>', '?'), ('<BANG>', '!')]

        # need to perform some preprocessing and do the embeddings
        print "loading dataset"
        with open(self.corpusfp, 'r') as f:
            self.dataset = f.readlines()
            self.dataset = ' '.join(self.dataset)

        print "converting dataset to list of sentences"
        self.sentences = self.dataset.translate(None, '\t\n').lower()

        # convert sentence-ending punctuation to words and append a '.' to each
        self.sentences = replaceAll(self.sentences, self.puncToTag)

        # split the corpus into a list of sentences
        self.sentences = self.sentences.split('.')

        # remove any other punctuation and convert to lower
        self.sentences = [sentence[1:].translate(None, "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~").split() for sentence in self.sentences]

        self.model = gensim.models.Word2Vec(self.sentences, min_count=5, size=50, workers=4)

        a = self.model.wv.vocab
        # dictionary of possible characters
        # self.chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
        #               't', 'u', 'v', 'w', 'x', 'y', 'z', \
        #               '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '.', ',', '!', '?', '(', ')', '\'', '"',
        #               ' ']
        # self.num_chars = len(self.chars)

        self.num_words = len(self.vocabulary)

        # dictionary mapping characters to indices
        self.word2idx = {word: i for (i, word) in enumerate(self.vocabulary)}
        self.idx2word = {i: word for (i, word) in enumerate(self.vocabulary)}

        '''
        #training portion of language model
        '''

        # input sequence of character indices
        self.input = tf.placeholder(tf.int32, [1, seq_len])

        # convert to one hot
        one_hot = tf.one_hot(self.input, self.num_words)

        # rnn layer
        self.gru = GRUCell(rnn_size)
        outputs, states = tf.nn.dynamic_rnn(self.gru, one_hot, sequence_length=[self.seq_len], dtype=tf.float32)
        outputs = tf.squeeze(outputs, [0])

        # ignore all outputs during first read steps
        outputs = outputs[first_read:-1]

        # softmax logit to predict next character (actual softmax is applied in cross entropy function)
        logits = tf.layers.dense(outputs, self.num_words, None, True, tf.orthogonal_initializer(), name='dense')

        # target character at each step (after first read chars) is following character
        targets = one_hot[0, first_read + 1:]

        # loss and train functions
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        self.optimizer = tf.train.AdamOptimizer(0.0002, 0.9, 0.999).minimize(self.loss)

        '''
        #generation portion of language model
        '''

        # use output and state from last word in training sequence
        state = tf.expand_dims(states[-1], 0)
        output = one_hot[:, -1]

        # save predicted characters to list
        self.predictions = []

        # generate 100 new characters that come after input sequence
        for i in range(100):
            # run GRU cell and softmax
            output, state = self.gru(output, state)
            logits = tf.layers.dense(output, self.num_words, None, True, tf.orthogonal_initializer(), name='dense',
                                     reuse=True)

            # get index of most probable character
            output = tf.argmax(tf.nn.softmax(logits), 1)

            # save predicted character to list
            self.predictions.append(output)

            # one hot and cast to float for GRU API
            output = tf.cast(tf.one_hot(output, self.num_words), tf.float32)

        # init op
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, text, iterations=100000):
        '''
        train network on given text

        parameters:
          - text: string
            string to train network on
          - iterations: int (default: 100000)
            number of iterations to train for

        outputs:
            None
        '''

        # convert characters to indices
        print "converting text in indices"
        text_indices = [self.word2idx[char] for char in text if char in self.word2idx]

        # get length of text
        text_len = len(text_indices)

        # train
        for i in range(iterations):

            # select random starting point in text
            start = np.random.randint(text_len - self.seq_len)
            sequence = text_indices[start:start + self.seq_len]

            # train
            feed_dict = {self.input: [sequence]}
            loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            sys.stdout.write("iterations %i loss: %f  \r" % (i + 1, loss))
            sys.stdout.flush()

            # show generated sample every 100 iterations
            if (i + 1) % 100 == 0:
                feed_dict = {self.input: [sequence]}
                pred = self.sess.run(self.predictions, feed_dict=feed_dict)
                sample = ''.join([self.idx2char[idx[0]] for idx in pred])
                print "iteration %i generated sample: %s" % (i + 1, sample)


if __name__ == "__main__":
    import re

    # load sample text
    with open('corpus-large.txt', 'r') as f:
        text = f.read()

    # clean up text
    text = text.replace("\n", " ")  # remove linebreaks
    text = re.sub(' +', ' ', text)  # remove duplicate spaces
    text = text.lower()  # lowercase

    # train rnn
    rnn = character_rnn('corpus-large.txt')
    rnn.train(text)
