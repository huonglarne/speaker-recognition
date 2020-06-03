import os
import pickle
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture
from FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")

class ModelsTrainer:

    def __init__(self, train_path):
        self.train_path = [os.path.join(train_path, name) for name in os.listdir(train_path)]
        self.features_extractor = FeaturesExtractor()

    def process(self):
        for i, path in enumerate(self.train_path):
            files = [os.path.join(path, name) for name in os.listdir(path)]
            features = self.collect_features(files)
            speaker_gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
            speaker_gmm.fit(features)
            self.save_gmm(speaker_gmm, "Speaker"+str(i))

    def collect_features(self, files):
        features = np.asarray(())
        # extract features for each speaker
        for file in files:
            print("%5s %10s" % ("PROCESSNG ", file))
            # extract MFCC & delta MFCC features from audio
            vector    = self.features_extractor.extract_features(file)
            # stack the features
            if features.size == 0:  features = vector
            else: features = np.vstack((features, vector))
        return features

    def save_gmm(self, gmm, name):
        filename = name + ".gmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVING", filename,))

if __name__== "__main__":
    models_trainer = ModelsTrainer("D:\pythonprojects\signal\dsp-speaker-recognition\data\Seg_train_audio")
    models_trainer.process()
