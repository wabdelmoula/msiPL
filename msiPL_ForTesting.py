# -*- coding: utf-8 -*-
"""
Implementation of msiPL (Abdelmoula et al): Ultra-fast Test Data Analysis

    Our trained msiPL model is applied on new unseen test data which was withheld
    from a large 3D MSI datacube. Foe the Analysis of 3D MSI data, msiPL provides:
    - Ultra-fast Analysis (just a few seconds)
    - Memory efficient: unlike conventional methods there is no need to load 
      the full complex 3D MSI at once into the RAM. 
                        
"""

import numpy as np
np.random.seed(1337)
from tensorflow import set_random_seed
set_random_seed(2)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import time


# ========= Load MSI Data without prior peak picking (hdf5 format) ==========
f =  h5py.File('Test_Data/MouseKindey_z73.h5','r')  
MSI_test = f["Data"]  
All_mz = f["mzArray"]  
nSpecFeatures = len(All_mz)
xLocation = np.array(f["xLocation"]).astype(int)
yLocation = np.array(f["yLocation"]).astype(int)
col = max(np.unique(xLocation))
row = max(np.unique(yLocation))
im = np.zeros((col,row))
mzId = np.argmin(np.abs(All_mz[:] - 6227.9))
for i in range(len(xLocation)):
    im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = MSI_test[i,mzId] #image index starts at 0 not 1
plt.imshow(im);plt.colorbar()

# ====== Load VAE_BN: a fully-connected neural network model ========
from Computational_Model import *
input_shape = (nSpecFeatures, )
intermediate_dim = 512
latent_dim = 5
VAE_BN_Model = VAE_BN(nSpecFeatures,  intermediate_dim, latent_dim)
myModel, encoder = VAE_BN_Model.get_architecture()
myModel.summary()

# ================ Load The Trained Model =====================
myModel.load_weights('TrainedModel_Kidney_Z1.h5')


# *****************************************************************************
# ****************** Ultra Fast Analysis on new unseen data *****************

# ============= 1. Manifold Learning and  Model Predictions ===============
start_time = time.time()
encoded_imgs = encoder.predict(MSI_test) # Learned non-linear manifold
decoded_imgs = myModel.predict(MSI_test) # Reconstructed Data
print("--- %s seconds : Ultra-Fast, isn't it?" % (time.time() - start_time))
dec_TIC = np.sum(decoded_imgs, axis=-1)

# ======= 2. Compare Original and Reconstructed (inferred) Data ========
mse = mean_squared_error(MSI_test,decoded_imgs)
meanSpec_Rec = np.mean(decoded_imgs,axis=0) 
print('mean squared error(mse)  = ', mse)
meanSpec_Orig = np.mean(MSI_test,axis=0) # TIC-norm original MSI Data
N_DecImg = decoded_imgs/dec_TIC[:,None]  # TIC-norm reconstructed MSI  Data
meanSpec_RecTIC = np.mean(N_DecImg,axis=0)
plt.plot(All_mz,meanSpec_Orig); plt.plot(All_mz,meanSpec_RecTIC,color = [1.0, 0.5, 0.25]); 
plt.title('TIC-norm distribution of average spectrum: Original and Predicted')


# ======== 3. Model Parameters of the Latent Space ==========
Latent_mean, Latent_var, Latent_z = encoded_imgs

# ======== 4. Non-linear dimensionality Reduction  ==========
ndim = Latent_z.shape[1]
plt.figure(figsize=(14, 14))
for j in range(ndim):
    for i in range(len(xLocation)):
        im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = Latent_z[i,j]  
    ax = plt.subplot(1, ndim, j + 1)    
    plt.imshow(im,cmap="hot");  # plt.colorbar()   
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
 
# ========= 5. Visualize Original & Reconstructed (inferred) m/z images ==========
mzs = [2489.6,6627.9,8981.4,13961.2]  
directory = 'Results\\test'          
if not os.path.exists(directory):
    os.makedirs(directory)    
for indx in range(0,len(mzs)):
    mzId = np.argmin(np.abs(All_mz[:] - mzs[indx]))     
    for i in range(len(xLocation)):
        im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = N_DecImg[i,mzId] # Reconstructed TIC-norm m/z image
#        im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = MSI_test[i,mzId] # Original TIC-norm m/z image
    ax = plt.subplot(1, len(mzs), indx + 1)    
    plt.imshow(im);  # plt.colorbar()   
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imsave(directory + '\\mz' + str(All_mz[mzId]) + '_Rec.jpg',im)
#    plt.imsave(directory + '\\mz' + str(All_mz[mzId]) + '_Orig.jpg',im)

# *****************************************************************************
#********************* 6. Peak Learning (Manuscript Equation#4) ***************    
# Statistical Analysis on the trained neural network hyperparameter(weight)
from LearnPeaks import *
W_enc = encoder.get_weights()
# Normalize Weights by multiplying it with std of original data variables
std_spectra = np.std(MSI_test, axis=0) 
Beta = 2.5
Learned_mzBins, Learned_mzPeaks, mzBin_Indx, Real_PeakIdx = LearnPeaks(All_mz, W_enc,std_spectra,latent_dim,Beta,meanSpec_Orig)



# *****************************************************************************
# ========= Color Map ==============                                      
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# *********************** Downstream Data Analysis ****************************
# Data Clustering using GMM: applied on the encoded fetaures "Latent_z" 
# Peak Localization for each cluster 
nClusters = 7
gmm = GaussianMixture(n_components=nClusters,random_state=0).fit(Latent_z)
labels = gmm.predict(Latent_z)
labels +=1 # To Avoid confilict with the natural background value of 0
for i in range(len(xLocation)):
    im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = labels[i]
MyCmap = discrete_cmap(nClusters+1, 'jet')
plt.imshow(im,cmap=MyCmap);
plt.colorbar(ticks=np.arange(0,nClusters+1,1))
plt.axis('off')


# ======= Select a cluster of interest and correlate with the Learned_mzPeaks ===============
# 1. Select CLuster:
cluster_id = 6
Kimg = labels==cluster_id
Kimg = Kimg.astype(int)

for i in range(len(xLocation)):
    im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = Kimg[i]
segCmp = [MyCmap(0),MyCmap(cluster_id)]
cm = LinearSegmentedColormap.from_list('Walid_cmp',segCmp,N=2)
plt.imshow(im, cmap=cm);
plt.axis('off')

# 2. Correlate the Select CLuster with the Learned_mzPeaks:
Peaks_ID = [np.argmin(np.abs(All_mz[:] - Learned_mzPeaks[i])) for i in  range(len(Learned_mzPeaks))]
MSI_PeakList = MSI_test[:,Peaks_ID[:]] # get only MSI data only for the shotlisted learned m/z peaks
Corr_Val =  np.zeros(len(Learned_mzPeaks))
for i in range(len(Learned_mzPeaks)):
    Corr_Val[i] = stats.pearsonr(Kimg,MSI_PeakList[:,i])[0]
id_mzCorr = np.argmax(Corr_Val)
rank_ij =  np.argsort(Corr_Val)[::-1]

for i in range(len(xLocation)):
    im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = MSI_PeakList[i,id_mzCorr]  
plt.imshow(im)
plt.axis('off')
print('m/z', Learned_mzPeaks[id_mzCorr])
print('corr_Value = ', Corr_Val[id_mzCorr])

plt.plot(Learned_mzPeaks,Corr_Val)
print(['%0.4f' % i for i in Learned_mzPeaks[rank_ij[0:10]]])
print('Correlation Top 10 Ranked peaks:', end='')
print(['%0.4f' % i for i in Corr_Val[rank_ij[0:10]]])  
