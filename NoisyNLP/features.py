from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

from gensim.models import word2vec
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import AgglomerativeClustering

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

import regex as re
import subprocess
import string
import os
import sys
import six

from utils import *


class RegexFeatures(object):
    PATTERNS = {
        "isInitCapitalWord": re.compile(r'^[A-Z][a-z]+'),
        "isAllCapitalWord": re.compile(r'^[A-Z][A-Z]+$'),
        "isAllSmallCase": re.compile(r'^[a-z]+$'),
        "isWord": re.compile(r'^[a-zA-Z][a-zA-Z]+$'),
        "isAlphaNumeric": re.compile(r'^\p{Alnum}+$'),
        "isSingleCapLetter": re.compile(r'^[A-Z]$'),
        "containsDashes": re.compile(r'.*--.*'),
        "containsDash": re.compile(r'.*\-.*'),
        "singlePunctuation": re.compile(r'^\p{Punct}$'),
        "repeatedPunctuation": re.compile(r'^[\.\,!\?"\':;_\-]{2,}$'),
        "singleDot": re.compile(r'[.]'),
        "singleComma": re.compile(r'[,]'),
        "singleQuote": re.compile(r'[\']'),
        "isSpecialCharacter": re.compile(r'^[#;:\-/<>\'\"()&]$'),
        "fourDigits": re.compile(r'^\d\d\d\d$'),
        "isDigits": re.compile(r'^\d+$'),
        "isNumber": re.compile(r'^((\p{N}{,2}([,]?\p{N}{3})+)(\.\p{N}+)?)$'),
        "containsDigit": re.compile(r'.*\d+.*'),
        "endsWithDot": re.compile(r'\p{Alnum}+\.$'),
        "isURL": re.compile(r'^http[s]?://'),
        "isMention": re.compile(r'^(RT)?@[\p{Alnum}_]+$'),
        "isHashtag": re.compile(r'^#\p{Alnum}+$'),
        "isMoney": re.compile(r'^\$((\p{N}{,2}([,]?\p{N}{3})+)(\.\p{N}+)?)$'),
    }
    def __init__(self):
        print("Initialized RegexFeature")
    @staticmethod
    def process(word):
        features = dict()
        for k, p in six.iteritems(RegexFeatures.PATTERNS):
            if p.match(word):
                features[k] = True
        return features
    

def preprocess_token(x, to_lower=False):
    for k in ENTITY_MAPPINGS.keys():
        if RegexFeatures.PATTERNS[k].match(x):
            return ENTITY_MAPPINGS[k]
    if to_lower:
        x = x.lower()
    return x

    
    
WORD_SPLITTER = re.compile(r'[\p{Punct}\s]+')
class DictionaryFeatures(object):
    def __init__(self, dictDir):
        self.word2dictionaries = {}
        self.word2hashtagdictionaries = {}
        self.dictionaries = []
        i = 0
        for d in os.listdir(dictDir):
            print("read dict %s"%d, file=sys.stderr)
            self.dictionaries.append(d)
            if d == '.svn':
                continue
            for line in open(dictDir + "/" + d):
                word = line.rstrip('\n')
                word = word.strip(' ').lower()
                word = WORD_SPLITTER.sub(" ", word)
                word_hashtag = "".join(WORD_SPLITTER.split(word))
                if word not in self.word2dictionaries:
                    self.word2dictionaries[word] = str(i)
                else:   
                    self.word2dictionaries[word] += "\t%s" % i
                if word_hashtag not in self.word2hashtagdictionaries:
                    self.word2hashtagdictionaries[word_hashtag] = str(i)
                else:
                    self.word2hashtagdictionaries[word_hashtag] += "\t%s" % i
            i += 1
    
    MAX_WINDOW_SIZE=6
    def GetDictFeatures(self, words, i):
        features = []
        phrase = ' '.join(words[i:i+1]).lower().strip(string.punctuation)
        phrase = WORD_SPLITTER.sub(" ", phrase)
        if phrase in self.word2dictionaries:
            for j in self.word2dictionaries[phrase].split('\t'):
                features.append('DICT=%s' % self.dictionaries[int(j)])
        for window in range(1, self.MAX_WINDOW_SIZE+1):
            ## Forward
            start=i
            end =i + window + 1
            if start > -1 and end < len(words) + 1:
                phrase = ' '.join(words[start:end]).lower().strip(string.punctuation)
                phrase = WORD_SPLITTER.sub(" ", phrase)
                if phrase in self.word2dictionaries:
                    for j in self.word2dictionaries[phrase].split('\t'):
                        features.append('DICTFWD[+%s]=%s' % (window, self.dictionaries[int(j)]))
            ## Backward
            start = i - window
            end =i+1
            if start > -1 and end < len(words) + 1:
                phrase = ' '.join(words[start:end]).lower().strip(string.punctuation)
                phrase = WORD_SPLITTER.sub(" ", phrase)
                if phrase in self.word2dictionaries:
                    for j in self.word2dictionaries[phrase].split('\t'):
                        features.append('DICTBCK[-%s]=%s' % (window, self.dictionaries[int(j)]))
            ## Window        
            start = i - window
            end =i+window+1
            if start > -1 and end < len(words) + 1:
                phrase = ' '.join(words[start:end]).lower().strip(string.punctuation)
                phrase = WORD_SPLITTER.sub(" ", phrase)
                if phrase in self.word2dictionaries:
                    for j in self.word2dictionaries[phrase].split('\t'):
                        features.append('DICTWIN[%s]=%s' % (window, self.dictionaries[int(j)]))
                        
        """
        for window in range(1,self.MAX_WINDOW_SIZE):
            start=max(i-window+1, 0)
            end = start + window
            phrase = ' '.join(words[start:end]).lower().strip(string.punctuation)
            phrase = WORD_SPLITTER.sub(" ", phrase)
            if phrase in self.word2dictionaries:
                for j in self.word2dictionaries[phrase].split('\t'):
                    features.append('DICT=%s' % self.dictionaries[int(j)])
                    if window > 1:
                        features.append('DICTWIN[%s]=%s' % (window, self.dictionaries[int(j)]))
                        
        """                
        return list(set(features))
    
    def GetHashtagDictFeatures(self, word):
        features = []
        if len(word) < 2 or word[0] != "#":
            return features
        word = word[1:].lower().strip(string.punctuation)
        if word in self.word2hashtagdictionaries:
            for j in self.word2hashtagdictionaries[word].split('\t'):
                features.append('DICT_HASHTAG=%s' % self.dictionaries[int(j)])
        return list(set(features))

    


class WordVectors(object):
    def __init__(self, sentences, wordvec_file=None, enrich_iters=5):
        self.model = word2vec.Word2Vec(sentences,
                                       size=200, window=10, sg=1, hs=0, min_count=1,
                                       negative=10, workers=-1, iter=20)
        if wordvec_file is not None:
            self.model.intersect_word2vec_format(wordvec_file, binary=False)
        if enrich_iters > 0:
            for i in six.moves.range(enrich_iters):
                self.model.train(sentences, total_examples=len(sentences), epochs=self.model.iter)
        self.model.init_sims(replace=True)
        
    
    def get_clusters(self, n_clusters=50):
        self.cluster_model_ = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")
        cluster_ids = self.cluster_model_.fit_predict(self.model.syn0norm)
        self.cluster_mappings = {
            k: cluster_ids[v.index] 
            for k,v in six.iteritems(self.model.vocab)
                                }
        return self.cluster_mappings
    

class ClusterFeatures(object):
    def __init__(self,  cluster_dir, cluster_type="brown", n_clusters=100):
        self.cluster_dir = cluster_dir
        self.cluster_type = cluster_type
        self.n_clusters = n_clusters
        #self.cluster_vocab = None
        self.exec_paths = dict()
        self.cluster_file_path = None
        
    def set_exec_path(self, path):
        self.exec_paths[self.cluster_type]=path
        
    def set_cluster_file_path(self, path=None):
        if path is None:
            if self.cluster_type == "brown":
                path = "%s/paths" % (self.cluster_dir)
            elif self.cluster_type == "clark":
                path = "%s/clark_clusters.%s.txt" % (self.cluster_dir,
                                                     self.n_clusters)
        self.cluster_file_path = path
        
    def gen_training_data(self, sentences, filename):
        with open(filename, "w", encoding="utf-8") as fp:
            for seq in sentences:
                if self.cluster_type == "brown":
                    print(" ".join(seq), file=fp)
                elif self.cluster_type == "clark":
                    print("\n".join(seq), file=fp)
                    print("\n", file=fp)
                else:
                    raise("Error: incorrect cluster type")
    
    def _gen_brown_clusters(self, input_data_path):
        """
        ! /home/entity/Downloads/brown-cluster/wcluster --text all_sequences.txt --c 100 --output_dir word_clusters
        """
        self.cluster_file_path = "%s/paths" % (self.cluster_dir)
        commands = [self.exec_paths[self.cluster_type],
                    "--text", input_data_path,
                    "--c", self.n_clusters,
                    "--output_dir", self.cluster_dir]
        
        return commands
        
    def _gen_clark_clusters(self, input_data_path):
        """
        ! /home/entity/Downloads/clark_pos_induction/src/bin/cluster_neyessenmorph -s 10 -m 1 -i 10 -x all_sequences.clark.txt all_sequences.clark.txt 32 > all_sequences.clark_clusters.32.txt 2> clark.err
        """
        self.cluster_file_path = "%s/clark_clusters.%s.txt" % (self.cluster_dir, self.n_clusters)
        commands = [self.exec_paths[self.cluster_type],
                    "-s", 10, "-m", 1, "-i", 10,
                    "-x", input_data_path,
                    input_data_path, self.n_clusters,
                    ">", self.cluster_file_path]
        return commands
    
    def gen_clusters(self,  input_data_path, output_dir_path, n_clusters=None):
        self.cluster_dir=output_dir_path
        if n_clusters is None:
            n_clusters = self.n_clusters
        self.n_clusters = n_clusters
        if self.cluster_type == "brown":
            commands = self._gen_brown_clusters(input_data_path)
        elif self.cluster_type == "clark":
            commands = self._gen_clark_clusters(input_data_path)
        cmd = " ".join(str(c) for c in commands)
        subprocess.check_call(cmd, shell=True)
        
        
    def read_clusters(self):
        if self.cluster_type == "brown":
            return self._read_brown_clusters()
        elif self.cluster_type == "clark":
            return self._read_clark_clusters()
    
    
    
    
    def _read_brown_clusters(self):
        cluster_vocab=dict()
        with open(self.cluster_file_path) as fp:
            for line in fp:
                cid, word, counts = line.strip().split("\t")
                cluster_vocab[word] = cid
        return cluster_vocab
    
    def _read_clark_clusters(self):
        cluster_vocab=dict()
        with open(self.cluster_file_path) as fp:
            for line in fp:
                try:
                    word, cid, prob = line.strip().split(" ")
                    cluster_vocab[word] = (cid, float(prob))
                except:
                    ## Skipping line
                    pass
        return cluster_vocab
        
        
#### Predict tweet type


class GlobalFeatures(object):
    
    def __init__(self, word2vec_model=None, cluster_vocabs=None,
                       dict_features=None, cat_names=None, WORD_IDX=0):
        self.word2vec_model = word2vec_model
        self.cluster_vocabs = cluster_vocabs
        self.dict_features = dict_features
        self.WORD_IDX = WORD_IDX
        self.cat_names = cat_names
        
        
    def get_global_sequence_features(self, sent, predictions=None):
        features = dict()
        sent_length = len(sent) * 1.
        for word in sent:
            word = word[self.WORD_IDX]
            lookup_key = preprocess_token(word, to_lower=True)
            if self.word2vec_model and lookup_key in self.word2vec_model:
                for i,v in enumerate(self.word2vec_model[lookup_key]):
                    features["_GLOBAL_WORDVEC_%s" % i] = dict.get(features, "_GLOBAL_WORDVEC_%s" % i, 0) + v
            if self.cluster_vocabs and lookup_key in self.cluster_vocabs:
                v = dict.get(self.cluster_vocabs, lookup_key)
                features["_GLOBAL_CLUSTER_=%s" % v] = dict.get(features, "_GLOBAL_CLUSTER_=%s" % v, 0) + 1
        features = {k: v / sent_length for k,v in six.iteritems(features)}
        if predictions:
            for k, prob in six.iteritems(predictions):
                features["_MODEL_=%s" % k] = prob
        return [features for word in sent]
        
    
    def tweet_features(self, sent):
        features = {}
        sent_length = len(sent) * 1.
        for widx, word in enumerate(sent):
            word = word[self.WORD_IDX]
            lookup_key = preprocess_token(word, to_lower=True)
            if self.word2vec_model and lookup_key in self.word2vec_model:
                for i,v in enumerate(self.word2vec_model[lookup_key]):
                    features["_GLOBAL_WORDVEC_%s" % i] = dict.get(features, "_GLOBAL_WORDVEC_%s" % i, 0) + v
            if self.cluster_vocabs and lookup_key in self.cluster_vocabs:
                v = dict.get(self.cluster_vocabs, lookup_key)
                features["_GLOBAL_CLUSTER_=%s" % v] = dict.get(features, "_GLOBAL_CLUSTER_=%s" % v, 0) + 1
            if self.dict_features:
                d_features = self.dict_features.GetDictFeatures([k[WORD_IDX] for k in sent], widx)
                for k in d_features:
                    features[k] = dict.get(features, k, 0) + 1
                d_hashtag_features = self.dict_features.GetHashtagDictFeatures(word)
                for k in d_hashtag_features:
                    features[k] = dict.get(features, k, 0) + 1
        #features = {k: v / sent_length for k,v in six.iteritems(features)}
        return features
            
    def get_sequence_features(self, sequences):
        features = [self.tweet_features(sent) for sent in sequences]
        return features
    
    def is_tweet_type(self, sent, cat_type):
        for t in sent:
            if t.tag != "O":
                if t.tag[2:] == cat_type:
                    return 1
        return 0
    
    def fit_feature_dict(self, sequences):
        train_data = self.get_sequence_features(sequences)
        self.feature2matrix = DictVectorizer()
        self.feature2matrix.fit(train_data)
        
    def tranform_sequence2feature(self, sequences):
        train_data = self.get_sequence_features(sequences)
        return self.feature2matrix.transform(train_data)
   
    def fit_model(self, train_sequences, test_sequences=None):
        if test_sequences is None:
            test_sequences = train_sequences
        self.fit_feature_dict(train_sequences)
        tweet_X_train = self.tranform_sequence2feature(train_sequences)
        tweet_X_test = self.tranform_sequence2feature(test_sequences)
        self.models = dict()
        for cat_type in self.cat_names:
            print("Processing: %s" % cat_type)
            y_train = np.array([self.is_tweet_type(sent, cat_type) for sent in train_sequences])
            y_test = np.array([self.is_tweet_type(sent, cat_type) for sent in test_sequences])
            model = LogisticRegression(solver="lbfgs", multi_class="multinomial")
            model.fit(tweet_X_train, y_train)
            y_pred = model.predict(tweet_X_test)
            print(classification_report(y_test, y_pred))
            self.models[cat_type] = model
            
            
    def get_global_predictions(self, sequences):
        predictions = {}
        X_train = self.tranform_sequence2feature(sequences)
        for k, model in six.iteritems(self.models):
            y_pred = model.predict_proba(X_train)[:, 1]
            predictions[k] = y_pred
        keys = predictions.keys()
        predictions = [dict(zip(keys, v)) for v in zip(*predictions.values())]
        return predictions  
    
        
        
        
#### MODEL FEATURES


def get_word_form(word, vocab=None, lower=False):
    if lower:
        word = word.lower()
    if vocab:
        vocab_search_word = word.lower() 
        word = "OOV" if vocab_search_word not in vocab else word
    return word

def get_clust_tag_value(lookup_key, cluster_vocab, cluster_tag,
                       clust_values_tuple=False):
    v = cluster_vocab[lookup_key]
    if clust_values_tuple:
        cluster_tag = "{}={}".format(cluster_tag, v[0])
        v = v[1]
    return (cluster_tag, v)    


def get_word(sent, widx, WORD_IDX):
    if WORD_IDX is None:
        return sent[widx]
    return sent[widx][WORD_IDX]


def gen_cluster_features(sent, widx, cid, lookup_key, cluster_vocab, WORD_IDX=0, 
                         cluster_tag="_CLUST_", dropout=0, clust_values_tuple=False,
                        interactions=False):
    features = {}
    if lookup_key in cluster_vocab:
        center_word_f = get_clust_tag_value(lookup_key, cluster_vocab,
                                            cluster_tag, clust_values_tuple=clust_values_tuple)
        features.setdefault(*center_word_f)
        ## Previous word
        if widx > 0:
            lookup_key_prev = preprocess_token(get_word(sent, widx-1, WORD_IDX), to_lower=True)
            if lookup_key_prev in cluster_vocab:
                prev_word_f = get_clust_tag_value(lookup_key_prev, cluster_vocab, 
                                                 "{}[-1]".format(cluster_tag),
                                                  clust_values_tuple=clust_values_tuple)
                features.setdefault(*prev_word_f)
                if interactions:
                    if np.random.rand() > dropout:
                        if clust_values_tuple:
                            features["{}|{}".format(
                                    center_word_f[0], prev_word_f[0]
                                )] = (center_word_f[1]*prev_word_f[1])
                        else:
                            features["{}={}|{}={}".format(*(center_word_f+prev_word_f))] = True
        ## Next word
        if widx < len(sent) -1:
            lookup_key_next = preprocess_token(get_word(sent, widx+1, WORD_IDX), to_lower=True)
            if lookup_key_next in cluster_vocab:
                next_word_f = get_clust_tag_value(lookup_key_next, cluster_vocab, 
                                                 "{}[+1]".format(cluster_tag),
                                                 clust_values_tuple=clust_values_tuple)
                features.setdefault(*next_word_f)
                if interactions:
                    if np.random.rand() > dropout:
                        if clust_values_tuple:
                            features["{}|{}".format(
                                    next_word_f[0], center_word_f[0]
                                )] = (next_word_f[1]*center_word_f[1])
                        else:
                            features["{}={}|{}={}".format(*(next_word_f+center_word_f))] = True
        """
        for k,v in six.iteritems(features):
            if isinstance(v, float) or isinstance(v, bool) or isinstance(v, str) or isinstance(v,int):
                continue
            else:
                print features
                print k,v
                raise
        """
        return features
    

def word2features(sent, widx, WORD_IDX=0,
                  extra_features={},
                  vocab=None, dict_features=None,
                  vocab_presence_only=False,
                  dict_interactions=False,
                  interactions=False,
                  lowercase=True,
                  window=0,
                 verbose=False, dropout=0.5,
                 word2vec_model=None,
                 cluster_vocabs=None):
    word = get_word(sent, widx, WORD_IDX)
    features = {
        'bias': True,
        #'word_normed': word.lower(),
        #'suffix_3': word[-3:].lower(),
        #'suffix_2': word[-2:].lower(),
        #'prefix_3': word[-3:].lower(),
        #'prefix_2': word[-2:].lower(),
    }
    ## Word2Vec or Brown Cluster features
    if word2vec_model or cluster_vocabs:
        lookup_key = preprocess_token(word, to_lower=True)
        ## Word2Vec features
        if word2vec_model and lookup_key in word2vec_model:
            word2vec_features = {"_WORDVEC_%s" % i: v for i,v in enumerate(word2vec_model[lookup_key])}
            features.update(word2vec_features)
        ## Brown cluster features
        if cluster_vocabs:
            if not isinstance(cluster_vocabs, list):
                cluster_vocabs = [cluster_vocabs]
            for cid, cluster_vocab in enumerate(cluster_vocabs):
                cluster_type="BROWN"
                if cluster_vocab:
                    clust_values_tuple=False
                    if isinstance(six.next(six.itervalues(cluster_vocab)), tuple):
                        clust_values_tuple = True
                        cluster_type="CLARK"
                    cluster_tag = "__{}_CLUSTER_{}__".format(cluster_type, cid)
                    clust_features = gen_cluster_features(
                        sent, widx, cid, lookup_key, cluster_vocab, WORD_IDX=WORD_IDX,
                        cluster_tag=cluster_tag, dropout=dropout, 
                        clust_values_tuple=clust_values_tuple, interactions=interactions)
                    if clust_features:
                        features.update(clust_features)

    ## Vocab Feature
    if vocab:
        if vocab_presence_only:
            features["word_normed"] = word.lower() in vocab
        else:
            features["word_normed"] = word.lower() if word.lower() in vocab else "OOV"
    ## Regex Feature
    regex_features = RegexFeatures.process(word)
    features.update(regex_features)
    if interactions:
        if widx > 0:
            regex_features_prev = RegexFeatures.process(get_word(sent, widx-1, WORD_IDX))
            features.update(("%s[-1]" % k, v) for k,v in six.iteritems(regex_features_prev))
            features.update({
                    ("%s[-1]|%s" % (k1,k), True)
                    for k, v in six.iteritems(regex_features)
                    for k1, v1 in six.iteritems(regex_features_prev)
                    if v & v1 & (np.random.rand() > dropout)
                })
        if widx < len(sent)-1:
            regex_features_next = RegexFeatures.process(get_word(sent, widx+1, WORD_IDX))
            features.update(("%s[+1]" % k, v) for k,v in six.iteritems(regex_features_next))
            features.update({
                    ("%s|%s[+1]" % (k,k1), True)
                    for k, v in six.iteritems(regex_features)
                    for k1, v1 in six.iteritems(regex_features_next)
                    if v & v1 & (np.random.rand() > dropout)
                })
    ## Gazetteer Feature
    if dict_features:
        if WORD_IDX is None:
            words = sent
        else:
            words = [k[WORD_IDX] for k in sent]
        d_features = dict_features.GetDictFeatures(words, widx)
        features.update({k: True for k in d_features})
        d_features = sorted(d_features)
        features.update({"%s|%s" % (k,k1): True for i, k in enumerate(d_features)
                         for k1 in d_features[i+1:]
                         if np.random.rand() > dropout})
        d_hashtag_features = dict_features.GetHashtagDictFeatures(word)
        features.update({k: True for k in d_hashtag_features})
        d_hashtag_features = sorted(d_hashtag_features)
        features.update({"%s|%s" % (k,k1): True for i, k in enumerate(d_hashtag_features)
                         for k1 in d_hashtag_features[i+1:]
                         if np.random.rand() > dropout})
    ## Extra features
    features.update(extra_features)
    ## Start Feature
    if widx == 0:
        features['__BOS__'] = True
    ## End Feature
    if widx == len(sent)-1:
        features['__EOS__'] = True
    ## Word Feature
    if not lowercase:
        if np.random.rand() > dropout:
            features["word_original"] = word
    curr_word_normed = get_word_form(word, lower=lowercase, vocab=vocab)
    #features["normed_word"] = curr_word_normed
    
    """
    if widx > 0:
        prev_word = get_word_form(sent[widx-1][WORD_IDX], lower=lowercase)
        features["word[-1]"] = prev_word
        if interactions:
            if np.random.rand() > dropout:
                features["word[-1]=%s|word=%s" % (prev_word, curr_word_normed)] = True
    if widx < len(sent) -1:
        next_word = get_word_form(sent[widx+1][WORD_IDX], lower=lowercase)
        features["word[+1]"] = next_word
        if interactions:
            if np.random.rand() > dropout:
                features["word[+1]=%s|word=%s" % (next_word, curr_word_normed)] = True
    """
    return features

def sent2features(sent, extra_features=None, **kwargs):
    if extra_features:
        return [word2features(sent, i, extra_features=extra_features[i],
                              **kwargs) for i in range(len(sent))]
    return [word2features(sent, i, **kwargs) for i in range(len(sent))]

def sent2labels(sent, lbl_id=1):
    return [k[lbl_id] for k in sent]





