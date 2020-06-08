# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:21:03 2016
@author: ORCHISAMA
@modified by: LTHOANGG
"""

from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUDistance
from mel_coefficients import mfcc
from LPC import lpc
from train import training
import os
import speech_recognition as sr
import wave
import pickle

students = ["Dao Duong Hoang Long", "Ngo Xuan Minh", "Nguyen Chi Thanh", "Le Trong Hoang", "Trinh Thao Phuong", "Nguyen Lan Huong", "Nguyen Le Thanh Ha", "Nguyen Huy An", "Dang Anh Duc"]
directory = os.getcwd() + '/test'
nSpeaker = len(next(os.walk(directory))[2]) - 1
nfiltbank = 12
orderLPC = 20
train = int(input("Do you want to re-train models?\n1.Yes\n2.No\nYour command: "))
if train == 1:
    (codebooks_mfcc, codebooks_lpc) = training(nfiltbank, orderLPC)
    pickle.dump(codebooks_mfcc, open("models/mfcc_model.sav" , 'wb'))
    pickle.dump(codebooks_lpc, open("models/lpc_model.sav" ,  'wb'))
else:
    codebooks_mfcc = pickle.load(open("models/mfcc_model.sav", 'rb'))
    codebooks_lpc = pickle.load(open("models/lpc_model.sav", 'rb'))
fname = str()
nCorrect_MFCC = 0
nCorrect_LPC = 0


def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k,:,:])
        dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0]) 
        if dist < distmin:
            distmin = dist
            speaker = k
            
    return speaker
    
f = open(os.getcwd()+"/test/filename.txt", "r")
filename = f.read()
for i in range(nSpeaker):
    fname = '/'+ filename + str(i+1) + '.wav'
    print('Now speaker ', str(i+1), 'features are being tested')
    (fs,s) = read(directory + fname)
    mel_coefs = mfcc(s,fs,nfiltbank)
    lpc_coefs = lpc(s, fs, orderLPC)
    sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
    sp_lpc = minDistance(lpc_coefs, codebooks_lpc)
    
    print('Speaker ', (i+1), ' in test matches with speaker ', (sp_mfcc+1), ' in train for training with MFCC')
    print('Speaker ', (i+1), ' in test matches with speaker ', (sp_lpc+1), ' in train for training with LPC')
   
    if i == sp_mfcc:
        nCorrect_MFCC += 1
    if i == sp_lpc:
        nCorrect_LPC += 1
    

percentageCorrect_MFCC = (nCorrect_MFCC/nSpeaker)*100
print('Accuracy of result for training with MFCC is ', percentageCorrect_MFCC, '%')
percentageCorrect_LPC = (nCorrect_LPC/nSpeaker)*100
print('Accuracy of result for training with LPC is ', percentageCorrect_LPC, '%')

ques = int(input("Do you want to live testing?\n1. Yes\n2. No\nYour command: "))
while ques == 1:
    recording = sr.Recognizer()
    with sr.Microphone() as mic:
        print("BOT: I'm listening...")
        audio = recording.listen(mic)
        try:
            you = recording.recognize_google(audio)
        except Exception:
            print("Speak again!!!")
            continue
        if "bye" in you:
            ques = 0
            print("See you again!")
            break
        elif you == "":
            pass
        else:
            
            file_audio = open("livetesting/yours.wav", "wb")
            file_audio.write(audio.get_wav_data())
            file_audio.close()
            print("You said: " + you)
            fname = "/livetesting/yours.wav"
            (fs,s) = read(os.getcwd() + fname)
            mel_coefs = mfcc(s,fs,nfiltbank)
            lpc_coefs = lpc(s, fs, orderLPC)
            sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
            sp_lpc = minDistance(lpc_coefs, codebooks_lpc)
            print("You are speaker:\n - " + str(sp_mfcc+1) + " tested by MFCC\n - " + str(sp_lpc+1) + " tested by LPC.")
            print("You are:\n - " + students[sp_mfcc+1] + " tested by MFCC\n - " + students[sp_lpc+1] + " tested by LPC.")
