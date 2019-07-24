import os
import pickle
import six
import sys
from features import sent2features, DictionaryFeatures, ClusterFeatures, WordVectors
from utils import load_sequences, process_glovevectors

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

TRAIN_FILE = os.path.join(DATA_DIR, "wnut16_train")
DEV_FILE = os.path.join(DATA_DIR, "wnut16_dev")
HEGE_TRAIN_FILE = os.path.join(DATA_DIR, "hege.test.tsv")

TWITTER_NER_MODEL_FILE = "twitter_ner%s_model.pkl"

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DICTIONARY_DIR = os.path.join(DATA_DIR, "cleaned", "custom_lexicons")
WORDVEC_FILE_RAW = os.path.join(DATA_DIR, "glove.twitter.27B.200d.txt")
WORDVEC_FILE_PROCESSED = os.path.join(DATA_DIR, "glove.twitter.27B.200d.txt.processed.txt")
GIMPLE_TWITTER_BROWN_CLUSTERS_DIR = os.path.join(DATA_DIR, "50mpaths2")
TEST_ENRICHED_DATA_BROWN_CLUSTER_DIR = os.path.join(DATA_DIR, "brown_clusters%s")
TEST_ENRICHED_DATA_CLARK_CLUSTER_DIR = os.path.join(DATA_DIR, "clark_clusters%s")

class TwitterNER:
    def __init__(self, training_data_name="_wnut_and_hege",
                 train_files=[(TRAIN_FILE, "utf-8"), (HEGE_TRAIN_FILE, "utf-8")]):
        model_dir = os.path.join(MODEL_DIR, "python%s" % sys.version_info.major)
        with open(os.path.join(model_dir, TWITTER_NER_MODEL_FILE % training_data_name), "rb") as pickle_file:
            self.model = pickle.load(pickle_file)

        self.dict_features = DictionaryFeatures(DICTIONARY_DIR)

        all_sequences = load_sequences(DEV_FILE)
        for (train_file, encoding) in train_files:
            all_sequences.extend(load_sequences(train_file, sep="\t", encoding=encoding))
        all_tokens = [[t[0] for t in seq] for seq in all_sequences]

        if not os.path.exists(WORDVEC_FILE_PROCESSED):
            process_glovevectors(WORDVEC_FILE_RAW)
        self.word2vec_model = WordVectors(all_tokens, WORDVEC_FILE_PROCESSED)

        gimple_brown_cf = ClusterFeatures(GIMPLE_TWITTER_BROWN_CLUSTERS_DIR, cluster_type="brown")
        gimple_brown_cf.set_cluster_file_path(GIMPLE_TWITTER_BROWN_CLUSTERS_DIR)
        self.gimple_brown_clusters = gimple_brown_cf.read_clusters()

        test_enriched_data_brown_cluster_dir = TEST_ENRICHED_DATA_BROWN_CLUSTER_DIR % training_data_name
        test_enriched_data_brown_cf = ClusterFeatures(test_enriched_data_brown_cluster_dir,
                                                      cluster_type="brown", n_clusters=100)
        test_enriched_data_brown_cf.set_cluster_file_path()
        self.test_enriched_data_brown_clusters = test_enriched_data_brown_cf.read_clusters()

        test_enriched_data_clark_cluster_dir = TEST_ENRICHED_DATA_CLARK_CLUSTER_DIR % training_data_name
        test_enriched_data_clark_cf = ClusterFeatures(test_enriched_data_clark_cluster_dir,
                                                      cluster_type="clark", n_clusters=32)
        test_enriched_data_clark_cf.set_cluster_file_path()
        self.test_enriched_data_clark_clusters = test_enriched_data_clark_cf.read_clusters()

    def get_features(self, tokens):
        return sent2features(tokens, WORD_IDX=None, vocab=None,
                             dict_features=self.dict_features, vocab_presence_only=False,
                             window=4, interactions=True, dict_interactions=True,
                             lowercase=False, dropout=0,
                             word2vec_model=self.word2vec_model.model,
                             cluster_vocabs=[
                               self.gimple_brown_clusters,
                               self.test_enriched_data_brown_clusters,
                               self.test_enriched_data_clark_clusters
                             ])

    def get_entities(self, tokens):
        predictions = self.model.predict([self.get_features(tokens)])

        entities = []
        previous_state = None
        entity_start = None
        for i in six.moves.range(len(tokens)):
            token = tokens[i]
            label = predictions[0][i]
            state = label[0]
            if state in ("B", "U") or \
               (state in ("I", "E") and previous_state not in ("B", "I")):
                entity_start = i
            if state in ("E", "U") or \
               (state in ("B", "I") and (i == len(tokens) - 1 or predictions[0][i + 1][0] not in ("I", "E"))):
                entity_type = label[2:]
                if entity_type is not None:
                    entities.append((entity_start, i + 1, entity_type))
                entity_start = None
            previous_state = state
        return entities
