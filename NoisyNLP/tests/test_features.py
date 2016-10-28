from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

import unittest

from NoisyNLP.features import *

class FeatruesTest(unittest.TestCase):
    def test_regex_features(self):
        regex_features = RegexFeatures()
        output = {u'containsDigit': True, u'isAlphaNumeric': True}
        output_f = regex_features.process("ABC123")
        self.assertEqual(output_f, output)
        
    def test_dict_features(self):
        dict_features = DictionaryFeatures("./data/test/lexicons")
        output = [u'DICT=cap.1000']
        output_f = dict_features.GetDictFeatures(["new", "york", "is", "a", "great", "place", "in", "america"], 1)
        self.assertEqual(output_f, output)

    def test_hashtagdict_features(self):
        dict_features = DictionaryFeatures("./data/test/lexicons")
        output = [u'DICT_HASHTAG=cap.1000']
        output_f = dict_features.GetHashtagDictFeatures("#york")
        self.assertEqual(output_f, output)
        
    def test_wordvec_features(self):
        sequences = [("the", "man", "crossed", "the", "road", "in", "new", "york"),
             ("new", "york", "is", "a", "great", "place", "in", "america")
            ]
        word2vec = WordVectors(sequences, "./data/test/glove.twitter.200d.txt.processed.txt")
        output = set(sum(sequences, ()))
        output_f = set(word2vec.model.vocab.keys())
        self.assertEqual(output_f, output)
        
        
    def test_wordvec_clusters(self):
        sequences = [("the", "man", "crossed", "the", "road", "in", "new", "york"),
             ("new", "york", "is", "a", "great", "place", "in", "america")
            ]
        wv_model = WordVectors(sequences, "./data/test/glove.twitter.200d.txt.processed.txt")
        clusters = wv_model.get_clusters(n_clusters=3)
        output = set(clusters.values())
        output_f = set(range(3))
        self.assertEqual(output_f, output)
        
    def test_brown_clusters(self):
        cf = ClusterFeatures("data/test/brown_clusters/", cluster_type="brown")
        cf.set_cluster_file_path()
        clusters = cf.read_clusters()
        self.assertEqual(clusters[u'15mins'], u'00000')
        
    def test_brown_clusters(self):
        cf = ClusterFeatures("data/test/clark_clusters/", cluster_type="clark", n_clusters=32)
        cf.set_cluster_file_path()
        clusters = cf.read_clusters()
        self.assertEqual(clusters[u'and'], (u'27', 0.249616))