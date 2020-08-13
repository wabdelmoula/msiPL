# -*- coding: utf-8 -*-
"""
Implementation of msiPL (Abdelmoula et al): Identify an informative peak Peaks

    - This function should be called after training the model
    - Briefly: this is a backpropagated-based threshold analysis on the neural network weight hyper-parameter (see, Equation#4)
      This analysis is to identify m/z contributed strongly to the learned non-manifold (encoded structures).
"""
import numpy as np
from scipy.signal import argrelextrema


def LearnPeaks(All_mz, W_enc, std_spectra, latent_dim,Beta,meanSpec_Orig):
    W1 = W_enc[0] # Connected with the input layer
    W2 = W_enc[6] # z_mean Layer
    for EncID in range(latent_dim):
        W2_EncFeat1 = W2[:,EncID]          
        Act_Neuron_W2 = np.argsort(-W2_EncFeat1) #Note: -ve is used to sort descending
        W2_EncFeat1[Act_Neuron_W2[0]]
        Neuron_W1 = W1[:,Act_Neuron_W2[0]]
        Weights_norm_W1 = std_spectra*Neuron_W1
        ij =  np.argsort(Weights_norm_W1)[::-1]
        Weights_norm_W1 = np.sort(Weights_norm_W1)[::-1]
        Weights_norm_W1[0]
        
    # ======== Threshold Weights mean + Beta*std:
        T = np.mean(Weights_norm_W1) + Beta*np.std(Weights_norm_W1)
        PeakID = ij[np.argwhere(Weights_norm_W1 >= T)]; PeakID = PeakID[:,0] #Ranked indices
        
    # ======== Get union list of m/z from all encFetaures ========
        Enc_mz = [All_mz[i] for i in PeakID]
        if EncID==0:
            Learned_mzBins = []
            Common_PeakID = []
        Learned_mzBins = list(set().union(Enc_mz , Learned_mzBins))
        Common_PeakID = list(set().union(PeakID , Common_PeakID))
        
        if EncID==latent_dim-1:
            Learned_mzBins = np.sort(Learned_mzBins)
            Common_PeakID = np.sort(Common_PeakID)
        
    LocalMax = np.squeeze(np.transpose(argrelextrema(meanSpec_Orig, np.greater))) 
    mz_LocalMax = [All_mz[i] for i in LocalMax]
    Nearest_Peakindx = [np.argmin(np.abs(mz_LocalMax[:] - Learned_mzBins[i])) for i in  range(len(Learned_mzBins))]
    Peak_Indx = np.unique(Nearest_Peakindx)
    Learned_mzPeaks = [mz_LocalMax[i] for i in Peak_Indx]
    Learned_mzPeaks = np.asarray(Learned_mzPeaks)
    
    Real_PeakIdx = [np.argmin(np.abs(All_mz[:] - Learned_mzPeaks[i])) for i in  range(len(Learned_mzPeaks))]


    return Learned_mzBins, Learned_mzPeaks, Common_PeakID,Real_PeakIdx 

