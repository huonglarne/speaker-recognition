# Data

I did a little preprocess for the audio files. I cut silent segments out of the original file and then separated them into smaller chunks.

# Code

The code is a modified version of [this project](https://github.com/SuperKogito/Voice-based-gender-recognition). Based on Dinh Anh's idea, I replaced the two-gender model with a nine-speaker one.

The model uses the Mel-Frequency Cepstrum Coefficients (MFCC) to extract features and the Gaussian Mixture Model (GMM) to evaluate the "relevance" of a test audio file to the deduced features.
