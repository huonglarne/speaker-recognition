# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:57:32 2016

@author: ORCHISAMA
@modified by: LTHOANGG
"""

from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import lbg
from mel_coefficients import mfcc
from LPC import lpc
import matplotlib.pyplot as plt
import os

f = open(os.getcwd()+"/train/filename.txt", "r")
filename = f.read()

def training(nfiltbank, orderLPC):
    directory = os.getcwd() + '/train'
    nSpeaker = len(next(os.walk(directory))[2]) - 1
    nCentroid = 16
    codebooks_mfcc = np.empty((nSpeaker,nfiltbank,nCentroid))
    codebooks_lpc = np.empty((nSpeaker, orderLPC, nCentroid))
    directory = os.getcwd() + '/train';
    fname = str()

    for i in range(nSpeaker):
        fname = '/' + filename + str(i+1) + '.wav'
        print('Now speaker ', str(i+1), 'features are being trained' )
        (fs,s) = read(directory + fname)
        mel_coeff = mfcc(s, fs, nfiltbank)
        lpc_coeff = lpc(s, fs, orderLPC)
        codebooks_mfcc[i,:,:] = lbg(mel_coeff, nCentroid)
        codebooks_lpc[i,:,:] = lbg(lpc_coeff, nCentroid)

    print('Training completed')
    
    #plotting 5th and 6th dimension MFCC features on a 2D plane
    #comment lines 54 to 71 if you don't want to see codebook
    codebooks = np.empty((2, nfiltbank, nCentroid))
    mel_coeff = np.empty((2, nfiltbank, 68))
   
    for i in range(2):
        fname = '/' + filename + str(i+2) + '.wav'
        (fs,s) = read(directory + fname)
        mel_coeff[i,:,:] = mfcc(s, fs, nfiltbank)[:,0:68]
        codebooks[i,:,:] = lbg(mel_coeff[i,:,:], nCentroid)
   
    
    return (codebooks_mfcc, codebooks_lpc)
    
    
