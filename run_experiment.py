from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

import time, datetime

from NoisyNLP.utils import *
from NoisyNLP.features import *
from NoisyNLP.models import *
from NoisyNLP.experiments import *

train_files = ["./data/cleaned/train.BIEOU.tsv"]
dev_files = ["./data/cleaned/dev.BIEOU.tsv", "./data/cleaned/dev_2015.BIEOU.tsv"]
test_files = ["./data/cleaned/test.BIEOU.tsv"]
vocab_file = "./vocab.no_extras.txt"
outdir = "./test_exp"

exp = Experiment(outdir, train_files, dev_files, test_files, vocab_file)
all_sequences = [[t[0] for t in seq] 
                        for seq in (exp.train_sequences + exp.dev_sequences + exp.test_sequences)]
wv_model = WordVectors(all_sequences,
                       "/home/entity/Downloads/GloVe/glove.twitter.27B.200d.txt.processed.txt")
word2vec_clusters = wv_model.get_clusters(n_clusters=50)
dict_features = DictionaryFeatures("./data/cleaned/custom_lexicons/")
exp.gen_model_data()
exp.fit_evaluate()
exp.describe_model()