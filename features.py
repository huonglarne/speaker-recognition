from python_speech_features import mfcc
import numpy as np

def get_feature(fs, signal):
    mfcc_feature = mfcc(signal, fs, nfft=4096)
    #mfcc_feature = mfcc(signal, fs)
    #print(fs)
    if len(mfcc_feature) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(signal)
    return mfcc_feature
