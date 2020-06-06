import numpy as np

from collections import defaultdict
from skgmm import GMMSet
from python_speech_features import mfcc, delta

import pickle
import time


def get_feature(fs, signal):
    """ Use MFCC to get features."""
    mfcc_feature = mfcc(signal, fs, nfft=2048)
    if len(mfcc_feature) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(signal)
    return mfcc_feature


class ModelInterface:
    """ Interface of `speaker-recognition.py`."""
    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()

    def enroll(self, name, fs, signal):
        feat = get_feature(fs, signal)
        self.features[name].extend(feat)

    def train(self):
        self.gmmset = GMMSet()
        start_time = time.time()
        for name, feats in self.features.items():
            try:
                self.gmmset.fit_new(feats, name)
            except Exception as e :
                print ("%s failed"%(name))
        print ("Training done, took", time.time() - start_time, "seconds.")

    def dump(self, fname):
        """ Dump all models to file."""
        self.gmmset.before_pickle()
        with open(fname, 'wb') as f:
            pickle.dump(self, f, -1)
        self.gmmset.after_pickle()

    def predict(self, fs, signal):
        """ Return a label (name)."""
        try:
            feat = get_feature(fs, signal)
        except Exception as e:
            print (e)
        return self.gmmset.predict_one(feat)

    @staticmethod
    def load(fname):
        """ load from a dumped model file"""
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            R.gmmset.after_pickle()
            return R
