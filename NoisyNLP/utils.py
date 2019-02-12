from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from io import open

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

import numpy as np
import pandas as pd

from collections import namedtuple
import regex as re
import subprocess

Tag = namedtuple("Tag", ["token", "tag"])

OTHER_ENTITIES_KEYS=("isHashtag", "isMention", "isURL", "isMoney", "isNumber", "repeatedPunctuation")
ENTITY_MAPPINGS={k: "__%s__" % k for k in OTHER_ENTITIES_KEYS}


def load_sequences(filename, sep="\t", notypes=False, test_data=False, encoding='utf-8'):
    sequences = []
    with open(filename, encoding=encoding) as fp:
        seq = []
        for line in fp:
            line = line.strip()
            if line:
                line = line.split(sep)
                if test_data:
                    assert len(line) == 1
                    line.append("?")
                if notypes:
                    line[1] = line[1][0]
                seq.append(Tag(*line))
            else:
                sequences.append(seq)
                seq = []
        if seq:
            sequences.append(seq)
    return sequences


def load_vocab(filename):
    vocab = set()
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            vocab.add(line)
    return vocab


def get_cat_names(sequences):
    cat_names = set()
    for seq in sequences:
        for t in seq:
            if t[1] != "O":
                cat_names.add(t[1][2:])
    return cat_names
            

    
def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr)) 
        
        
def plot_cm(y_test, y_pred, labels=[], axis=1):
    labels_s = dict((k,i) for i,k in enumerate(labels))
    cm = np.zeros((len(labels), len(labels)))
    for i,j in zip(sum(y_test, []), sum(y_pred, [])):
        i = labels_s[i]
        j = labels_s[j]
        cm[i,j] += 1
    with plt.rc_context(rc={'xtick.labelsize': 12, 'ytick.labelsize': 12,
                       'figure.figsize': (16,14)}):
        sns.heatmap(cm * 100/ cm.sum(axis=axis, keepdims=True),
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
    return cm


def print_sequences(sequences, predictions, filename, test_data=False, notypes=False):
    with open(filename, "w", encoding='utf-8') as fp:
        for seq, pred in zip(sequences, predictions):
            for t, p in zip(seq, pred):
                token, tag = t
                if tag[0] == "U":
                    tag = "B%s" % tag[1:]
                if tag[0] == "E":
                    tag = "I%s" % tag[1:]
                if p[0] == "U":
                    p = "B%s" % p[1:]
                if p[0] == "E":
                    p = "I%s" % p[1:]
                if notypes:
                    tag = tag[0]
                    p = p[0]
                if test_data:
                    line = "\t".join((token, p))
                else:
                    line = "\t".join((token, tag, p))
                print(line, file=fp)
            print("", file=fp)
            


RESULTS_REGEX=re.compile(r'\s+([a-z\-]*):.*?([0-9\.]+)%;.*?([0-9\.]+)%;.*?([0-9\.]+)\s+([0-9]+)')
def parse_results(results):
    results = results.splitlines()
    result_obj = dict()
    result_obj.update((k,v) for k,v in zip(("processed_tokens", "total", "found", "correct"),
                                          re.findall(r'[0-9]+', results[0].strip())))
    overall = re.match(r'.*?([0-9\.]+).*?([0-9\.]+)%;.*?([0-9\.]+)%;.*?\s+([0-9\.]+)', results[1]).groups()
    result_obj["overall_accuracy"] = float(overall[0])
    prfs_vals = [("overall",) + tuple(map(float, overall[1:] + (0.,)))]
    for line in results[2:]:
        if not line.strip():
            continue
        items = RESULTS_REGEX.match(line).groups()
        prfs_vals.append(items[:1] + tuple(map(float, items[1:])))
    result_obj["prfs"] = pd.DataFrame(prfs_vals, columns=["category", "precision", "recall", "F1", "support"])
    return result_obj
    
    
def print_results(result_obj):
    print("Processed %s tokens." % result_obj["processed_tokens"])
    print("Phrases: total=%s, found=%s, correct=%s" % (
            result_obj["total"],
            result_obj["found"],
            result_obj["correct"]
        ))
    print(("Overall accuracy: %.2f" % result_obj["overall_accuracy"]))
    print(result_obj["prfs"])
    
def get_conll_eval(filename):
    cmd = "cat \"%s\" | tr '\\t' ' ' | perl -ne '{chomp;s/\\r//g;print $_,\"\\n\";}' | python NoisyNLP/conlleval.py" % filename
    print("Running:\n%s" % cmd)
    results = subprocess.check_output(cmd, shell=True)
    return parse_results(results)
    
            
            

def print_regex_matches_all(sentences):
    other_entities = {
        k: [] for k in OTHER_ENTITIES_KEYS
    }
    
    for seq in sentences:
        for t in seq:
            for k in other_entities.keys():
                if RegexFeatures.PATTERNS[k].match(t):
                    other_entities[k].append(t)
    for k, v in other_entities.iteritems():
        print(k, len(v))
    
    
GLOVE_TWEET_MAPPINGS={
    "<user>": "isMention",
    "<hashtag>": "isHashtag",
    "<number>": "isDigit",
    "<url>": "isURL",
    "<allcaps>": "isAllCapitalWord",
}

def process_glovevectors(filename):
    words, dim, errors = 0, 0, 0
    with open(filename) as fp:
        while True:
            try:
                line = next(fp)
                if dim == 0:
                    dim = len(line.split(" ")) - 1
            except UnicodeDecodeError:
                errors += 1
                continue
            except StopIteration:
                break
            if len(line.split(" ")) != dim + 1:
                errors += 1
                continue
            words+= 1
    print("Words: {}, dim: {}, errors: {}".format(words, dim, errors))
    with open(filename) as fp, open("{}.processed.txt".format(filename), "w") as fp1:
        fp1.write(str(words) + u" " + str(dim) + u"\n")
        while True:
            try:
                line = next(fp)
                line = line.strip().split(" ", 1)
                line[0] = dict.get(GLOVE_TWEET_MAPPINGS, line[0], line[0])
                out_line = line[0] + u" " + line[1] + u"\n"
                if len(out_line.split(" ")) == dim + 1:
                    fp1.write(out_line)
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
            except StopIteration:
                break
    print("Done")
    
    
    
def classification_report_to_df(report):
    report_list = []
    for i, line in enumerate(report.split("\n")):
        if i == 0:
            report_list.append(["class", "precision", "recall", "f1-score", "support"])
        else:
            line = line.strip()
            if line:
                if line.startswith("avg"):
                    line = line.replace("avg / total", "avg/total")
                line = re.split(r'\s+', line)
                report_list.append(tuple(line))
    return pd.DataFrame(report_list[1:], columns=report_list[0])
