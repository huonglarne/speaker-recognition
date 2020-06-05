# Data preprocessing

The [VAD.py file](https://github.com/huonglarne/speaker-recognition/blob/base-gender-recognition/VAD.py) is used to detect voice activity and eliminate silent parts.
The [segment-audio.py file](https://github.com/huonglarne/speaker-recognition/blob/base-gender-recognition/segment-audio.py} is used to split audio into small chunks for easier training and testing.

# Code

The code is a modified version of [this project](https://github.com/SuperKogito/Voice-based-gender-recognition). Based on Dinh Anh's idea, I replaced the two-gender model with a nine-speaker one.

The model uses the Mel-Frequency Cepstrum Coefficients (MFCC) to extract features and the Gaussian Mixture Model (GMM) to evaluate the "relevance" of a test audio file to the deduced features.

How to use:

- Feed the Train data path to [this file](https://github.com/huonglarne/speaker-recognition/blob/base-gender-recognition/ModelsTrainer.py). It will return .gmm files representing the features of each person's voice.

- Feed the Test data path and the Model path (folder containing .gmm files) to [this file](https://github.com/huonglarne/speaker-recognition/blob/base-gender-recognition/Identifier.py)

