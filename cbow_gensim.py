#sample code implementing CBOW Word2Vec by Shang Gao

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

from string import maketrans   # Required to call maketrans function.


def replaceAll(s, d):
    for k, v in d:
        s = s.replace(k, v)

    return s

#logging setup
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load 20 newsgroups dataset
print "loading dataset"
with open("corpus-large.txt", 'r') as f:
    dataset = f.readlines()
    dataset = ' '.join(dataset)

# dataset = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes')).data
# dataset = ' '.join(dataset)
# dataset = unicodedata.normalize('NFKD', dataset).encode('ascii','ignore')

#convert dataset to list of sentences
print "converting dataset to list of sentences"
puncToTag = [('.', ' <STOP>.'), ('?', ' <QUEST>.'), ('!', ' <BANG>.')]
tagToPunc = [('<STOP>', '.'), ('<QUEST>', '?'), ('<BANG>', '!')]

sentences = dataset.translate(None, '\t\n').lower()

# convert sentence-ending punctuation to words and append a '.' to each
sentences = replaceAll(sentences, puncToTag)

sentences = sentences.split('.')
sentences = [sentence[1:].translate(None, "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~").split() for sentence in sentences]

#train word2vec
print "training word2vec"
model = gensim.models.Word2Vec(sentences, min_count=5, size=50, workers=4)

#get most common words
print "getting common words"
dataset = [item for sublist in sentences for item in sublist]
counts = collections.Counter(dataset).most_common(500)

#reduce embeddings to 2d using tsne
print "reducing embeddings to 2D"
embeddings = np.empty((500,50))
for i in range(500):
    embeddings[i,:] = model[counts[i][0]]
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=7500)
embeddings = tsne.fit_transform(embeddings)

#plot embeddings
print "plotting most common words"
fig, ax = plt.subplots(figsize=(30, 30))
for i in range(500):
    # print("{}, {}, {}".format(embeddings[i, 0], embeddings[i, 1], counts[i][0]))
    ax.scatter(embeddings[i,0],embeddings[i,1])
    ax.annotate(counts[i][0].decode('unicode-escape'), (embeddings[i,0],embeddings[i,1]))

#save to disk
plt.savefig('plot.png')
