#!/usr/bin/env python3

import os
import sys
import itertools
import glob
import argparse
from utils import read_wav
from interface import ModelInterface
import pyaudio
import wave
from scipy.io import wavfile
from logmm import reduce_noise
# import vad 

def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """ Wav files in each input directory will be labeled as the basename of the directory.
             Note that wildcard inputs should be *quoted*, and they will be sent to glob.glob module.
             Examples:
             Train (enroll a list of person named person*, and mary, with wav files under corresponding directories):
             ./speaker-recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out
             Predict (predict the speaker of all wav files):
             ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
             """
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "enroll", "predict" or "record"',
                       required=True)

    parser.add_argument('-i', '--input',
                       help='Input Files(to predict) or Directories(to enroll)',
                       required=True)

    parser.add_argument('-m', '--model',
                       help='Model file to save(in enroll) or use(in predict)',
                       required=True)

    ret = parser.parse_args()
    return ret


def task_enroll(input_dirs, output_model):
    """ Enroll data for model output.

    Parameters
    ----------
    input_dirs: str
        Input directory containg the train dataset.
    output_model: str
        Name of the output file.
    """
    # Breaking down paths
    m = ModelInterface()
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]

    files = []
    if len(dirs) == 0:
        print ("No valid directory found!")
        sys.exit(1)

    for d in dirs:
        label = os.path.basename(d.rstrip('/'))
        wavs = glob.glob(d + '/*.wav')

        if len(wavs) == 0:
            print ("No wav file found in %s"%(d))
            continue
        for wav in wavs:
            try:
                fs, signal = read_wav(wav)
                m.enroll(label, fs, signal)
                print("wav %s has been enrolled"%(wav))
            except Exception as e:
                print(wav + " error %s"%(e))

    # Train and store data as in labeled directory
    m.train()
    m.dump(output_model)


def task_predict(input_files, input_model):
    """ Predict speaker following model output.

    Parameters
    ----------
    input_file: str
        Input file of test data.
    output_model: str
        Name of the model used for identification.
    """
    m = ModelInterface.load(input_model)
    for f in glob.glob(os.path.expanduser(input_files)):
        fs, signal = read_wav(f)
        label, score = m.predict(fs, signal)
        print (f, '->', label, ", accuracy ->", score)


def task_record(input_files, input_model):
    filename = "recorded.wav"
    chunk = 1024 # set the chunk size of 1024 samples
    FORMAT = pyaudio.paInt16 # sample format
    channels = 1 # mono, change to 2 if you want stereo
    sample_rate = 44100
    record_seconds = 5
    p = pyaudio.PyAudio() # initialize PyAudio object
    # open stream object as input & output
    stream = p.open(format=FORMAT, channels=channels, rate=sample_rate, input=True, output=True, frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()  
    p.terminate() # terminate pyaudio object
    # save audio file

    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames)) # write the frames as bytes
    wf.close() # close the file

    reduce_noise('recorded.wav', input_files)
    # audio_file = r"record.wav"
    # vad.process_file(audio_file)
    # input_files=wave.open('recorded.wav','wb')
    m = ModelInterface.load(input_model)
    
    for f in glob.glob(os.path.expanduser(input_files)):
        fs, signal = read_wav(f)
        label, score = m.predict(fs, signal)
        print (f, '->', label, ", accuracy ->", score)


if __name__ == "__main__":
    global args
    args = get_args()

    task = args.task
    if task == 'enroll':
        task_enroll(args.input, args.model)
    elif task == 'predict':
        task_predict(args.input, args.model)
    elif task == "record":
        task_record(args.input, args.model)
    else:
        print ("No valid directory found!")
        sys.exit(1)