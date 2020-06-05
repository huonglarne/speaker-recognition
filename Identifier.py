import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")

class Identifier:

    def __init__(self, test_path, model_path):
        self.test_path = [os.path.join(test_path, name) for name in os.listdir(test_path)]
        self.label = [path[70] for path in self.test_path]

        self.accuracy                 = 0
        self.total_sample          = len(self.label)
        self.features_extractor    = FeaturesExtractor()

        # load models
        self.gmm_paths = [os.path.join(model_path, name) for name in os.listdir(model_path)]
        self.gmm = [pickle.load(open(path, 'rb')) for path in self.gmm_paths]

        self.test_results = []

    def find_accuracy(self):
        samples = len(self.label)
        compare = [self.test_results[i] == self.label[i] for i in range(samples)]
        self.accuracy = sum(compare)/samples
        return self.accuracy

    def process(self):
        # read the test directory and get the list of test audio files
        for file in self.test_path:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features(file)
            winner = self.identify(vector)
            index = self.gmm.index(winner)

            model_name = os.path.basename(self.gmm_paths[index])
            self.test_results.append(model_name[-5])

            print(model_name[-5])


    def identify(self, vector):
        scores = []
        for model in self.gmm:
            is_speaker_scores = np.array(model.score(vector))
            is_speaker_log_likelihood = is_speaker_scores.sum()
            scores.append(is_speaker_log_likelihood)
        t = scores.index(max(scores))
        return self.gmm[t]

if __name__== "__main__":
    test_path = r"D:\pythonprojects\signal\dsp-speaker-recognition\cut_vad\test"
    model_path = r"D:\pythonprojects\signal\dsp-speaker-recognition\gmm"
    identifier = Identifier(test_path, model_path)
    identifier.process()
    print(identifier.find_accuracy())
