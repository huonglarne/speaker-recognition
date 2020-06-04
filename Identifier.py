import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")

class Identifier:

    def __init__(self, test_path, model_path):
        self.test_path = [os.path.join(test_path, name) for name in os.listdir(test_path)]
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()

        # load models
        self.gmm_paths = [os.path.join(model_path, name) for name in os.listdir(model_path)]
        self.gmm = [pickle.load(open(path, 'rb')) for path in self.gmm_paths]

    def process(self):
        # read the test directory and get the list of test audio files
        for file in self.test_path:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features(file)
            winner = self.identify(vector)
            index = self.gmm.index(winner)

            print(os.path.basename(self.gmm_paths[index]))

    def identify(self, vector):
        best_score = 0
        best_model = self.gmm[0]
        for model in self.gmm:
            is_speaker_scores = np.array(model.score(vector))
            is_speaker_log_likelihood = is_speaker_scores.sum()
            if is_speaker_log_likelihood > best_score:
                best_score = is_speaker_log_likelihood
                best_model = model
        return best_model

if __name__== "__main__":
    test_path = "D:\pythonprojects\signal\dsp-speaker-recognition\data\Test_file"
    model_path = "D:\pythonprojects\signal\dsp-speaker-recognition\data\Gmm"
    identifier = Identifier(test_path, model_path)
    identifier.process()
