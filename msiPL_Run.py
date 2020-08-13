# -*- coding: utf-8 -*-
"""
Implementation of msiPL (Abdelmoula et al): Model Training
This is the main file to run
This implementation is based on the public AI platforms of Keras and Tensorflow.

How to run the code?
	1- "msiPL_Run.py" is the main file that you should run first. The file should be running in a sequential manner, and we have
	provided required comments for instructions and guidance. In this file you will be able to:
		1.1. Load a dataset.
		1.2. Load the computational neural network architecture (VAE_BN).
		1.3. Train the model.
		1.3. Non-linear manifold learning and data visualization (non-linear dimensionality reduction)
		1.4. Evaluate the learning quality by estimation and reconstruction of the original data
		1.5. Peak Learning learning (Equation#4): to get a smaller list of informative peaks.
		1.6. Perform data clustering (GMM): The number of clusters can be set by the users or automatically using the BIC method.
		1.7. Identify localized peaks within each cluster.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(1337)
from tensorflow import set_random_seed
set_random_seed(2)

import os
import h5py
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import time

# ========= Color Map ==============                                      
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# ========= Load MSI Data without prior peak picking (hdf5 format) ==========
""" The MSI data is loaded as hdf5 file to maintain efficiency.
	We have noticed high efficiency in memory usage and fast performance in accessing
	high dimensional data form huge files when dealing with HDF5 formats as oppose to the imzML.
You can convert the imzML to hdf5 using the following steps:
	- first install the python packages “h5py” and "imzML"
	- Load the imzML file to get: 
		a- spectral data, and let's say save it in a variable called  "Spec_Data", 
		b- spatial information for each spectrum and save it "XCoord" and "YCoord".
	- use the hhdf5 method called "create_dataset" to save your data in h5. For example:
		- myHF = h5py.file("myData.h5",'w')
		- myHF.create_dataset('Data', data=Spec_Data)
		- myHF. create_dataset('xLocation', data=XCoord)
		- After you finish, close your h5 file: "myHF.close()"
"""
f =  h5py.File('Training_Data/MouseKindey_z1.h5','r')  
MSI_train = f["Data"]  # spectral information.
All_mz = f["mzArray"]  
nSpecFeatures = len(All_mz)
if MSI_train.shape[1] != nSpecFeatures:
    MSI_train = np.transpose(MSI_train)
xLocation = np.array(f["xLocation"]).astype(int)
yLocation = np.array(f["yLocation"]).astype(int)
col = max(np.unique(xLocation))
row = max(np.unique(yLocation))
im = np.zeros((col,row))
mzId = np.argmin(np.abs(All_mz[:] - 6227.9))
for i in range(len(xLocation)):
    im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = MSI_train[i,mzId] #image index starts at 0 not 1
plt.imshow(im);plt.colorbar()

# ====== Load VAE_BN: a fully-connected neural network model ========
""" myModel represents the msiPL architecture """
from Computational_Model import *
input_shape = (nSpecFeatures, )
intermediate_dim = 512 # Size of the first hidden layer
# ---- dimensions of the latent space (i.e. encoded features)
ans = int(input('The default value of the latent space dimensions is 5, would you like to set a different value? Yes=1; No=0 …:'))
if ans == 1:
    latent_dim = int (input('Please set the new dimensions of the latent space = '))
else:
    latent_dim = 5
# -------------------------------------
# Compile the msiPL computational model
VAE_BN_Model = VAE_BN(nSpecFeatures,  intermediate_dim, latent_dim)
myModel, encoder = VAE_BN_Model.get_architecture()
myModel.summary()

# ============= Model Training =================
""" The training processes involves: 
	epochs: 100 iterations
	batch_size: a randomly-shuffled subset of 128 spectra is loaded at a time into the RAM 
	This phase will run faster if a GPU is utilized
 """
try:
    start_time = time.time()
    history = myModel.fit(MSI_train, epochs=100, batch_size=128, shuffle="batch")   
    plt.plot(history.history['loss'])
    plt.ylabel('loss'); plt.xlabel('epoch')
    print("--- %s seconds ---" % (time.time() - start_time))
    myModel.save_weights('TrainedModel_Kidney_Z1.h5')
except MemoryError as error:
    import psutil
    Memory_Information = psutil.virtual_memory()
    print('>>> There is a memory issue: and here are a few suggestions:')
    print('>>>>>> 1- Make sure that you are using  python 64-bit.')
    print('>>>>>> 2- use a lower value for the batch_size (default is 128).')
    print('**** Here is some information about your memory (MB):', Memory_Information)


# ============= Model Predictions ===============
encoded_imgs = encoder.predict(MSI_train) # Learned non-linear manifold
decoded_imgs = myModel.predict(MSI_train) # Reconstructed Data
dec_TIC = np.sum(decoded_imgs, axis=-1)

# ======= Calculate mse between orig & rec. data =====
""" The mean squared error (mse): 
	the mse is used to evaluate the quality of the reconstructed data"""
mse = mean_squared_error(MSI_train,decoded_imgs)
meanSpec_Rec = np.mean(decoded_imgs,axis=0) 
print('mean squared error(mse)  = ', mse)
meanSpec_Orig = np.mean(MSI_train,axis=0) # TIC-norm original MSI Data
N_DecImg = decoded_imgs/dec_TIC[:,None]  # TIC-norm reconstructed MSI  Data
meanSpec_RecTIC = np.mean(N_DecImg,axis=0)
plt.plot(All_mz,meanSpec_Orig); plt.plot(All_mz,meanSpec_RecTIC,color = [1.0, 0.5, 0.25]); 
plt.title('TIC-norm distribution of average spectrum: Original and Predicted')

# ======== Model Parameters of the Latent Space ==========
""" Capturing the learned latent variable:
encoded features (Latent_z), and its mean and variance"""
Latent_mean, Latent_var, Latent_z = encoded_imgs

# ======== Visualize encoded Features (learned non-linear spectral manifold) ==========
ndim = Latent_z.shape[1]
plt.figure(figsize=(14, 14))
for j in range(ndim):
    for i in range(len(xLocation)):
        im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = Latent_z[i,j]  
    ax = plt.subplot(1, ndim, j + 1)    
    plt.imshow(im,cmap="hot");  # plt.colorbar()   
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# ========= Visualize Original & Reconstructed m/z images ==========
mzs = [2489.6,6627.9,8981.4,13961.2]  
directory = 'Results'          
if not os.path.exists(directory):
    os.makedirs(directory)    
for indx in range(0,len(mzs)):
    mzId = np.argmin(np.abs(All_mz[:] - mzs[indx]))     
    for i in range(len(xLocation)):
        im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = N_DecImg[i,mzId] # Reconstructed TIC-norm m/z image
#        im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = MSI_train[i,mzId] # Original TIC-norm m/z image
    ax = plt.subplot(1, len(mzs), indx + 1)    
    plt.imshow(im);  # plt.colorbar()   
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imsave(directory + '\\mz' + str(All_mz[mzId]) + '_Rec.jpg',im)
#    plt.imsave(directory + '\\mz' + str(All_mz[mzId]) + '_Orig.jpg',im)

#********************* Peak Learning (Manuscript Equation#4) ********************    
""" Statistical Analysis on the trained neural network hyper-parameter(weight)
	See Equation (4) in the main manuscript)
"""
from LearnPeaks import *
W_enc = encoder.get_weights()
# Normalize Weights by multiplying it with std of original data variables
std_spectra = np.std(MSI_train, axis=0) 
Beta = 2.5 # This variable can be adjusted by the user. We have observed good performance within the range [1,2.5] 
Learned_mzBins, Learned_mzPeaks, mzBin_Indx, Real_PeakIdx = LearnPeaks(All_mz, W_enc,std_spectra,latent_dim,Beta,meanSpec_Orig)
# save results of learned peaks in excel sheet
directory = 'Results'
if not os.path.exists(directory):
    os.makedirs(directory_mz) 
import pandas as pd
df_1 = pd.DataFrame({'mz Peaks': Learned_mzPeaks})
df_1.to_excel(directory+'/'+'Peaks_.xlsx', engine='xlsxwriter' , sheet_name='Sheet1')

# ******************* Downstream Data Analysis **************************
""" Now the msiPL has been trained to learn a non-linear manifold, now the
clustering step can be efficiently applied.
Data Clustering using GMM: 
	- Applied on the encoded features "Latent_z" 
	- Peak Localization within each cluster 
	- nClusters: this is the number of clusters that need to be set before running the GMM.
	"nClusters" can be set manually or automatically suggested based on an optimization process using the BIC algorithm. 
 """
# ---- Bayesian Information Criterion (BIC) combined with the Kneedle algorithm for optimal model selection:
""" The total number of K-clusters will be automatically suggested.
	- Different GMM models will generated using different number of K-clusters (e.g. K varies between[3,20])
	- The BIC scores will be computed for each GMM model.
	- The Kneedle algorithm is applied on the BIC scores to identify the point of maximum curvature (knee point).
	- The knee point points to the best model and suggest the expected number of K-clusters.
"""
from kneed import KneeLocator
# covariance_type = {'full', 'spherical', 'diag', 'tied'}
cov_Type = 'full'
n_components = np.arange(3, 20)
models = [GaussianMixture(n, covariance_type=cov_Type, random_state=0).fit(Latent_z)
          for n in n_components]

BIC_Scores = [m.bic(Latent_z) for m in models]
kneedle_point = KneeLocator(n_components, BIC_Scores, curve='convex', direction='decreasing')
print('The suggested number of clusters = ', kneedle_point.knee)
Elbow_idx = np.where(BIC_Scores==kneedle_point.knee_y)[0]

from matplotlib.ticker import MaxNLocator
plt.plot(n_components, BIC_Scores,'-g', marker='o',markerfacecolor='blue',markeredgecolor='orange',
         markeredgewidth='2',markersize=10,markevery=Elbow_idx)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc='best')
plt.xlabel('Number of clusters');
plt.ylabel('BIC score');
plt.title('The suggested number of clusters = '+ np.str(kneedle_point.knee))
# plt.plot(n_components, [m.aic(Latent_z) for m in models], label='AIC')
# Ref Kneedle algorithm [V. Satopaa et al., international conference on distributed computing systems workshops. IEEE, 2011.]

# ======================== Apply GMM on Encoded Features ============= 
start_time_gmm = time.time()
nClusters = (kneedle_point.knee) # this variable is set automatically based on the BIC algorithm
# nClusters = 7 # this variable could be tuned by the user
gmm = GaussianMixture(n_components=nClusters,covariance_type=cov_Type,random_state=0).fit(Latent_z)
labels = gmm.predict(Latent_z)
labels +=1 # To Avoid conflict with the natural background value of 0

# Spatial Clusters Distribution:
for i in range(len(xLocation)):
    im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = labels[i]
MyCmap = discrete_cmap(nClusters+1, 'jet')
plt.imshow(im,cmap=MyCmap);
plt.colorbar(ticks=np.arange(0,nClusters+1,1))
plt.axis('off')
print("Clustering time =  %s seconds" % (time.time() - start_time_gmm))

# ======= Select a cluster of interest and correlate with the Learned_mzPeaks ===============
# 1. Select CLuster:
cluster_id = 2
Kimg = labels==cluster_id
Kimg = Kimg.astype(int)

for i in range(len(xLocation)):
    im[ np.asscalar(xLocation[i])-1, np.asscalar(yLocation[i])-1] = Kimg[i]
segCmp = [MyCmap(0),MyCmap(cluster_id)]
cm = LinearSegmentedColormap.from_list('Walid_cmp',segCmp,N=2)
plt.imshow(im, cmap=cm);
plt.axis('off')

# 2. Correlate the Select CLuster with the Learned_mzPeaks:
# Note: it will also be fast to correlate the cluster with All_mz Data
Peaks_ID = [np.argmin(np.abs(All_mz[:] - Learned_mzPeaks[i])) for i in  range(len(Learned_mzPeaks))]
MSI_PeakList = MSI_train[:,Peaks_ID[:]] # get only MSI data only for the shotlisted learned m/z peaks
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