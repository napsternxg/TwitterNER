from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

import unittest

from NoisyNLP.experiments import *

class ExperimentTest(unittest.TestCase):
    def test_experiment(self):
        train_files = ["./data/cleaned/dev.BIEOU.tsv"]
        dev_files = ["./data/cleaned/dev.BIEOU.tsv"]
        test_files = ["./data/cleaned/dev.BIEOU.tsv"]
        vocab_file = "./vocab.no_extras.txt"
        outdir = "./test_exp"
        exp = Experiment(outdir, train_files, dev_files, test_files, vocab_file)
        exp.gen_model_data()
        exp.fit_evaluate()
        exp.describe_model()
        self.assertEqual(1, 1)