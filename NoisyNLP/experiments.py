from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

import os, sys
import time, datetime

from NoisyNLP.utils import *
from NoisyNLP.features import *
from NoisyNLP.models import *


def file2sequences(files, sep="\t", notypes=False):
    return sum((load_sequences(f) for f in files), [])
    
def _base_get_X_y(sequences):
    X = [sent2features(s)
         for s in sequences]
    y = [sent2labels(s) for s in sequences]
    return X, y
    
def get_X_y(sequences, proc_func=_base_get_X_y, label="Data", **kwargs):
    start = time.time()
    X, y = proc_func(sequences, **kwargs)
    end = time.time()
    process_time = end - start
    print("%s feature generation took: %s" % (label, datetime.timedelta(seconds=process_time)))
    return X, y

def get_types_tag(notypes=False):
    return "notypes" if notypes else "bieou"

class Experiment(object):
    TRAIN_PRED_FILE="train.%s.tsv"
    DEV_PRED_FILE="dev.%s.tsv"
    TEST_PRED_FILE="test.%s.tsv"
    
    
    def __init__(self, outdir, train_files, dev_files=None, test_files=None, vocab_file=None,
                 sep="\t", notypes=False):
        if not os.path.isdir(outdir):
            print("Directory %s doesn't exist." % outdir, file=sys.stderr)
            os.makedirs(outdir)
            print("Directory %s created." % outdir, file=sys.stderr)
        self.outdir = outdir
        self.notypes = notypes
        self.train_sequences = file2sequences(train_files, sep=sep ,notypes=notypes)
        if dev_files is not None:
            self.dev_sequences = file2sequences(dev_files, sep=sep ,notypes=notypes)
        if test_files is not None:
            self.test_sequences = file2sequences(test_files, sep=sep ,notypes=notypes)
        self.vocab = load_vocab(vocab_file)
        
        
    def gen_model_data(self, proc_func=_base_get_X_y, **kwargs):
        self.X_train, self.y_train = get_X_y(
            self.train_sequences, proc_func=proc_func, label="Train", **kwargs)
        self.X_dev, self.y_dev = get_X_y(
            self.dev_sequences, proc_func=proc_func, label="Dev", **kwargs)
        self.X_test, self.y_test = get_X_y(
            self.test_sequences, proc_func=proc_func, label="Test", **kwargs)
        print("Train: %s, %s\nDev: %s, %s\nTest: %s, %s" % (len(self.X_train), len(self.y_train),
                                                            len(self.X_dev), len(self.y_dev),
                                                            len(self.X_test), len(self.y_test)))
        
        
    def fit_evaluate(self, notypes=None):
        if notypes is None:
            notypes = self.notypes
        self.model = CRFModel()
        self.model.fit(self.X_train, self.y_train)
        for X, y, sequences, label in zip((self.X_train, self.X_dev, self.X_test), 
                              (self.y_train, self.y_dev, self.y_test),
                              (self.train_sequences, self.dev_sequences, self.test_sequences),
                              ("train", "dev", "test")):
            y_pred = self.model.predict(X)
            print("Evaluating %s data" % label)
            type_tag = get_types_tag(notypes)
            filename = "%s/%s.%s.tsv" % (self.outdir, label, type_tag)
            print_sequences(sequences, y_pred, filename)
            result_obj = get_conll_eval(filename)
            print_results(result_obj)
            if not notypes:
                type_tag = get_types_tag(~notypes)
                filename = "%s/%s.%s.tsv" % (self.outdir, label, type_tag)
                print_sequences(sequences, y_pred, filename, notypes=~notypes)
                result_obj = get_conll_eval(filename)
                print_results(result_obj)
    
    def describe_model(self):
        self.model.print_transitions()
        self.model.print_state_features()


        
        