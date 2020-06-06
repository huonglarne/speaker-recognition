# Speaker Recognition

## Introduction
This is a simple design based on a [simple design](https://github.com/crouchred/speaker-recognition-py3) of [speaker-recognition](https://github.com/ppwwyyxx/speaker-recognition). The project suppose to be the final assignment of Digital Signal Processing course.

## Installation

### Prerequisites

[Python3](https://www.python.org/download/releases/3.0/) and [git](https://git-scm.com/) is required and [pip](https://pip.pypa.io/en/latest/) is recommended for the installation.

### Dependencies
The dependencies for the project are
- [numpy](https://numpy.org/)
- [python-speech-features](https://python-speech-features.readthedocs.io/en/latest/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [scipy](https://www.scipy.org/)

Per can choose which way of installing dependencies:
- By `pip` and using `requirement.txt`

    ```
    pip install -r requirements.txt
    ```
- Still by `pip` but not using `requirement.txt` (in case per have some of them already):

    ```
    pip install (dependency)
    E.g: pip install python_speech_features
    ```

### Install
Per can simply clone this repository
```
    git clone https://github.com/huonglarne/speaker-recognition.git
```

## Usage
From command line:
```
    python3 speaker-recognition.py -t TASK -i INPUT -m MODEL
```
Where:
- `TASK` is the task which the script supposed to do, either "enroll", "predict" or "record".
- `INPUT` input directories to "enroll" and input files to "predict". If the task is "record, write "record.wav"
- `MODEL` is a file to save the model when enroll and use to predict the speaker.