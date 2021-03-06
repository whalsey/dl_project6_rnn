import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import sys

from sklearn.neighbors import KNeighborsClassifier

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

class word_rnn(object):
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

    def __init__(self, corpusfp, seq_len=200, first_read=50, rnn_size=200, embedding_size = 500):

        self.seq_len = seq_len
        self.first_read = first_read
        self.embedding_size = embedding_size

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
            self.dataset = unicode(self.dataset, 'utf-8')

        self.dataset = unicodedata.normalize('NFKD', self.dataset).encode('ascii', 'ignore')

        print "converting dataset to list of sentences"
        self.sentences = self.dataset.translate(None, '\t\n').lower()

        # convert sentence-ending punctuation to words and append a '.' to each
        self.sentences = replaceAll(self.sentences, self.puncToTag)

        self.dataset = self.sentences.translate(None, "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~")

        # split the corpus into a list of sentences
        self.sentences = self.sentences.split('.')

        # remove any other punctuation and convert to lower
        self.sentences = [sentence[1:].translate(None, "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~").split() for sentence in self.sentences]

        self.model = gensim.models.Word2Vec(self.sentences, min_count=5, size=self.embedding_size, workers=4)

        # we have to have a way to recover the closest word to the ouput of the RNN
        # we will do this with a 1-NN model trained on the embeddings of the entire vocabulary
        a = self.model.wv.vocab

        X = []
        y = []
        for el in a:
            X.append(self.model[el])
            y.append(el)

        X = np.array(X)
        y = np.array(y)

        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn.fit(X, y)

        self.num_words = len(a)

        '''
        #training portion of language model
        '''

        # input sequence of word embeddings
        self.input = tf.placeholder(tf.float32, [1, self.seq_len, self.embedding_size])

        # rnn layer
        self.gru = GRUCell(rnn_size)
        outputs, states = tf.nn.dynamic_rnn(self.gru, self.input, sequence_length=[self.seq_len], dtype=tf.float32)
        outputs = tf.squeeze(outputs, [0])

        # ignore all outputs during first read steps
        outputs = outputs[first_read:-1]

        # softmax logit to predict next character (actual softmax is applied in cross entropy function)
        logits = tf.layers.dense(outputs, self.embedding_size, None, True, tf.orthogonal_initializer(), name='dense')

        # target character at each step (after first read chars) is following character
        targets = self.input[0, first_read + 1:]

        # loss and train functions
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets))
        self.optimizer = tf.train.AdamOptimizer(0.00005, 0.9, 0.999).minimize(self.loss)

        '''
        #generation portion of language model
        '''

        # use output and state from last word in training sequence
        state = tf.expand_dims(states[-1], 0)
        output = self.input[:, -1]

        # save predicted characters to list
        self.predictions = []

        # generate 100 new characters that come after input sequence
        for i in range(100):
            # run GRU cell and softmax
            output, state = self.gru(output, state)
            logits = tf.layers.dense(output, self.embedding_size, None, True, tf.orthogonal_initializer(), name='dense',
                                     reuse=True)

            activation = tf.nn.tanh(logits)

            # save predicted character to list
            self.predictions.append(activation)

            # one hot and cast to float for GRU API
            output = activation

        # init op
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, iterations=100000):
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
        print "converting text to embeddings"
        embeddings = [self.model[word] for word in text.split(' ') if word in self.model.wv.vocab]

        # scale embeddings to between 0 and 1
        tmp = np.array(embeddings)
        max_t = tmp.max()
        min_t = tmp.min()

        embeddings = (tmp - min_t) / (max_t - min_t)

        # get length of text
        text_len = len(embeddings)

        # train
        for i in range(iterations):

            # select random starting point in text
            start = np.random.randint(text_len - self.seq_len)
            sequence = embeddings[start:start + self.seq_len]

            # train
            feed_dict = {self.input: [sequence]}
            loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            sys.stdout.write("iterations %i loss: %f  \r" % (i + 1, loss))
            sys.stdout.flush()

            # show generated sample every 100 iterations
            if (i + 1) % 100 == 0:
                feed_dict = {self.input: [sequence]}
                pred = self.sess.run(self.predictions, feed_dict=feed_dict)
                pred = [q[0] for q in pred]
                output = self.knn.predict(pred)
                sample = ' '.join(output)
                print "iteration {} generated sample: {}".format(i + 1, sample)


if __name__ == "__main__":
    import re

    # load sample text
    with open('corpus-large.txt', 'r') as f:
        text = f.read()

    # train rnn
    rnn = word_rnn('corpus-large.txt')
    rnn.train()
