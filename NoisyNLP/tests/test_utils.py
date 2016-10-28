from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

import unittest

from NoisyNLP.utils import *

class UtilsTest(unittest.TestCase):
    def test_load_sequences(self):
        sequences = load_sequences("./data/cleaned/train.BIEOU.tsv", sep="\t", notypes=False)
        self.assertEqual(len(sequences), 2394)
        
        
    def test_parse_results(self):
        results = """processed 23050 tokens with 937 phrases; found: 748 phrases; correct: 343.
accuracy:  94.34%; precision:  45.86%; recall:  36.61%; FB1:  40.71
          company: precision:  46.94%; recall:  31.94%; FB1:  38.02  49
         facility: precision:  10.81%; recall:   8.70%; FB1:   9.64  37
          geo-loc: precision:  45.45%; recall:  58.64%; FB1:  51.21  209
            movie: precision:   0.00%; recall:   0.00%; FB1:   0.00  8
      musicartist: precision:  28.57%; recall:   3.51%; FB1:   6.25  7
            other: precision:  26.45%; recall:  17.88%; FB1:  21.33  121
           person: precision:  61.39%; recall:  65.16%; FB1:  63.22  259
          product: precision:  24.14%; recall:  15.22%; FB1:  18.67  29
       sportsteam: precision:  77.78%; recall:  20.00%; FB1:  31.82  27
           tvshow: precision:   0.00%; recall:   0.00%; FB1:   0.00  2
"""
        r = parse_results(results)
        self.assertEqual(r[u'overall_accuracy'], 94.34)
        
        results = """processed 46469 tokens with 1499 phrases; found: 5 phrases; correct: 0.
accuracy:  94.64%; precision:   0.00%; recall:   0.00%; FB1:   0.00
                 : precision:   0.00%; recall:   0.00%; FB1:   0.00  5
"""
        r = parse_results(results)
        self.assertEqual(r[u'overall_accuracy'], 94.64)
        
        
    def test_get_conll_eval(self):
        r = get_conll_eval("./test_wv.crf.bieou.tsv")
        self.assertEqual(r[u'overall_accuracy'], 91.84)
        
    def test_process_glove_vectors(self):
        process_glovevectors("./data/test/glove.twitter.200d.txt")
        self.assertEqual(1,1)


