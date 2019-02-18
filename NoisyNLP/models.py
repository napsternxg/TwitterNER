from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

from itertools import chain

from sklearn.metrics import make_scorer

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import regex as re
from collections import namedtuple, defaultdict, Counter, OrderedDict
from IPython.display import display
from joblib import load, dump, Parallel, delayed

import os, string, sys
import time, datetime
import subprocess

from utils import *

class CRFModel(object):
    def __init__(self, labels= None, skip_label="O", **kwargs):
        self.labels = labels
        self.skip_label = skip_label
        self.model = crf = sklearn_crfsuite.CRF(**kwargs)
        
    def fit(self, X_train, y_train):
        start = time.time()
        self.model.fit(X_train, y_train)
        end = time.time()
        process_time = end - start
        print("Model fitting took: %s" % datetime.timedelta(seconds=process_time))
        self._gen_labels()
        
    def _gen_labels(self):
        self.labels = list(self.model.classes_)
        self.labels.remove(self.skip_label)
        # group B and I results
        self.labels = sorted(
            self.labels, 
            key=lambda name: (name[1:], name[0])
        )
        self.boundary_labels = set([k[0] for k in self.labels]) - set([self.skip_label])
        self.category_labels = set([k[2:] for k in self.labels if k != self.skip_label])
        self.possible_labels = ["%s-%s" % (k1,k) for k in self.category_labels
            for k1 in self.boundary_labels] + [self.skip_label]

    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
        
    def evaluate(self, y_true, y_pred):
        print("Training accuracy: ", metrics.flat_f1_score(y_true, y_pred, 
                              average='weighted', labels=self.labels))
        report = metrics.flat_classification_report(
            y_true, y_pred, labels=self.labels, digits=3
        )
        return classification_report_to_df(report)
    
    
    def print_transitions(self, top_n=25):
        trans_features = Counter(self.model.transition_features_).most_common(top_n)
        print("Top %s likely transitions:" % top_n)
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        
    def print_state_features(self, top_n=10):
        state_features = Counter(self.model.state_features_).most_common(top_n)
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr)) 
            
    def show_transition_matrix(self):
        transition_matrix = np.zeros((len(self.possible_labels), len(self.possible_labels)))
        for k in self.possible_labels:
            for k1 in self.possible_labels:
                if (k,k1) in self.model.transition_features_:
                    i,j = self.possible_labels.index(k), self.possible_labels.index(k1)
                    transition_matrix[i,j] = self.model.transition_features_[(k,k1)]

        with plt.rc_context(rc={'xtick.labelsize': 10, 'ytick.labelsize': 10,
                                'font.size': 10,
                               'figure.figsize': (10,8)}):        
            ax = sns.heatmap(data=transition_matrix,
                    xticklabels=self.possible_labels, yticklabels=self.possible_labels,
                   cmap="RdGy")
        return ax

    def plot_cm(self, y_test, y_pred, axis=1):
        labels = self.labels + [self.skip_label]
        labels_s = dict((k,i) for i,k in enumerate(labels))
        cm = np.zeros((len(labels), len(labels)))
        for i,j in zip(sum(y_test, []), sum(y_pred, [])):
            i = labels_s[i]
            j = labels_s[j]
            cm[i,j] += 1
        with plt.rc_context(rc={'xtick.labelsize': 12, 'ytick.labelsize': 12,
                           'figure.figsize': (16,14)}):
            ax = sns.heatmap(cm * 100/ cm.sum(axis=axis, keepdims=True),
                        #cmap=sns.cubehelix_palette(n_colors=100, rot=-.4, as_cmap=True),
                        cmap="Greys",
                        xticklabels=labels,
                        yticklabels=labels)
            plt.ylabel("True labels")
            plt.xlabel("Predicted labels")
            title = "Precision Plot"
            if axis== 0:
                title = "Recall Plot"
            plt.title(title)
        return ax, cm
        
                
