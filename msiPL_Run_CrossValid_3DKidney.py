# -*- coding: utf-8 -*-
"""
Implementation of msiPL (Abdelmoula et al): Model Cross-Valiation Analysis

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
import matplotlib as mpl
import nibabel as nib
import pandas as pd
import time

# ======= Directory Information:
Cd = os.getcwd()
Bd = os.path.dirname(Cd)

# ========= Color Map ==============                                      
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# ====== Visualize Image: From 1D vector to Image ==============
def Image_Distribution(V,xLoc,yLoc):
    col = max(np.unique(xLoc))
    row = max(np.unique(yLoc))
    Myimg = np.zeros((col,row))
    for i in range(len(xLoc)):
        Myimg[np.asscalar(xLoc[i])-1, np.asscalar(yLoc[i])-1] = V[i]
    return Myimg

# ================= Correlate Cluster with MSI Data =============
def Correlate_Cluster_MSI(cluster_id,Labels,MSI_D,Peak_Indx,ZCoord_cv,XCoord_cv,YCoord_cv):
    Kimg = Labels==cluster_id
    Kimg = Kimg.astype(int)
    MSI_CleanPeaks = MSI_D[:,Peak_Indx[:]]
    Corr_Val =  np.zeros(len(Peak_Indx))
    
    for i in range(len(Peak_Indx)):
        Corr_Val[i] = stats.pearsonr(Kimg,MSI_CleanPeaks[:,i])[0]
    id_mzCorr = np.argmax(Corr_Val)
    rank_ij =  np.argsort(Corr_Val)[::-1]
    return Corr_Val, rank_ij, MSI_CleanPeaks

# ========================== 3D mz image ============================
def Get_3Dmz_nifti(MSI_CleanPeaks,mz_Peak,XCoord_cv,YCoord_cv,ZCoord_cv,directory):
    mzSections = np.unique(ZCoord_cv)
    Vol_mz = np.zeros((200,200,len(mzSections)))
    nSections = len(mzSections)
    directory_NIFT = directory + '\\mz_Vol\\Training'
    if not os.path.exists(directory_NIFT):
        os.makedirs(directory_NIFT)
    for Zsec in range(len(mzSections)):
        ij_r = np.argwhere(ZCoord_cv == mzSections[Zsec])
        indx = ij_r[:,0]
        xLoc = XCoord_cv[indx]
        yLoc = YCoord_cv[indx]
        MSI_2D = np.squeeze(MSI_CleanPeaks[indx])
        for idx in range(len(xLoc)):
            Vol_mz[np.asscalar(xLoc[idx])-1, np.asscalar(yLoc[idx])-1,Zsec] = MSI_2D[idx]  

    I_nii = nib.Nifti1Image(Vol_mz,affine=np.eye(4))
    nib.save(I_nii,directory_NIFT +'\\mz_' + str(mz_Peak) + '.nii')
    
#============= Spatial Distribution Encoded Fetaures =============
def get_EncFeatures(Latent_z,Train_idx,myZCoord,xLocation,yLocation,directory,order):
    myzSections = np.unique(myZCoord)
    ndim = Latent_z.shape[1]
    for zr in range(len(Train_idx)):
        ij_r = np.argwhere(myZCoord == myzSections[zr])
        indx = ij_r[:,0]
        xLoc = xLocation[indx]
        yLoc = yLocation[indx]
        zSection_Latent_z = np.squeeze(Latent_z[indx,])        
        plt.figure(figsize=(14, 14))
        for j in range(ndim):
            EncFeat = zSection_Latent_z[:,j] #encoded_imgs[i,0] #image index starts at 0 not 1 
            im = Image_Distribution(EncFeat,xLoc,yLoc);
            ax = plt.subplot(1, ndim, j + 1)    
            plt.imshow(im,cmap="hot");  # plt.colorbar()   
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
        directory_Latz = directory+'//Latent//Training_'+str(order)
        if not os.path.exists(directory_Latz):
            os.makedirs(directory_Latz)
        plt.savefig(directory_Latz + '\\EncFetaures_Tissue'+str(myzSections[zr])+'.png',bbox_inches='tight')
        
# ================== Get GMM Image for cv analysis ================        
def get_gmmImage(Train_idx,Features,nClusters,myZCoord,xLocation,yLocation,directory,order):
    myzSections = np.unique(myZCoord); Zsec=0; 
    C_imgs = np.zeros((200,200,len(range(1,len(Train_idx)+1,1)),nClusters))
    directoryGmm = directory+'//GMM//Training_'+str(order)
    if not os.path.exists(directoryGmm):
        os.makedirs(directoryGmm)
        
    for zr in range(len(Train_idx)):
        im = []
        ij_r = np.argwhere(myZCoord == myzSections[zr])
        indx = ij_r[:,0]
        xLoc = xLocation[indx]
        yLoc = yLocation[indx]
        zSection_labels = Features[indx]
        im = zSection_labels
        im = Image_Distribution(im,xLoc,yLoc);
        MyCmap = discrete_cmap(nClusters, 'jet')        
        plt.imshow(im,cmap=MyCmap);
        plt.colorbar(ticks=np.arange(0,nClusters,1))
        plt.axis('off')
        plt.show()        
        plt.imsave(directoryGmm + '\\gmm_Training_'+str(myzSections[zr])+'_K_' + str(nClusters) + '.png',im,cmap=MyCmap)  

         # Save single clusters:
        directory_SingleC =directoryGmm + '\\GMM_Section_'+str(myzSections[zr])
        if not os.path.exists(directory_SingleC):
            os.makedirs(directory_SingleC)
            
        for c in range(0,nClusters,1):
            cluster_id = c
            Kimg = zSection_labels[:]==cluster_id
            Kimg = Kimg.astype(int)
            for idx in range(len(xLoc)):
                C_imgs[np.asscalar(xLoc[idx])-1, np.asscalar(yLoc[idx])-1,Zsec,cluster_id] = Kimg[idx]  
            Kimg = Image_Distribution(Kimg,xLoc,yLoc);
            
            segCmp = [MyCmap(0),MyCmap(cluster_id)]
            segCmp[0]= (0,0,0)
            cm = LinearSegmentedColormap.from_list('Walid_cmp',segCmp,N=2)
            plt.imshow(Kimg, cmap=cm);
            plt.colorbar(ticks=np.arange(0,1,1))
            plt.axis('off')
            plt.show()
            plt.imsave(directory_SingleC + '\\Cluster_' + str(cluster_id) + '.png',Kimg,cmap=cm)  
        Zsec +=1
    #Save NIFTI
    directory_NIFT = directoryGmm + '\\NIFTI'
    if not os.path.exists(directory_NIFT):
        os.makedirs(directory_NIFT)
    for c in range(0,nClusters,1):
        I_nii = nib.Nifti1Image(C_imgs[:,:,:,c],affine=np.eye(4))
        nib.save(I_nii,directory_NIFT +'\\Label_' + str(c) + '.nii')

# ======== Save NIFTI image for each cluster ==================
def Cluster_To_Nifti(directory,C_imgs,nClusters):
    directory_NIFT = directory + '\\GMM_K'+str(nClusters)+'\\NIFTI'
    if not os.path.exists(directory_NIFT):
        os.makedirs(directory_NIFT) 
    for c in range(1,nClusters+1,1):
        I_nii = nib.Nifti1Image(C_imgs[:,:,:,c],affine=np.eye(4))
        nib.save(I_nii,directory_NIFT +'\\Label_' + str(c) + '.nii')
                  
# =================== Load 3D MSI Data ========================# 
Combined_MSI = []; XCoord = []; YCoord = []; ZCoord = []
TissueIDs = [x for x in range(1,74,1)]

for id in range(1,74,1):
    f =  h5py.File(Bd+'//hd5//MouseKindey_z' + str(id) + '.h5','r')
    MSI_train = f["Data"]
    mzList = f["mzArray"]
    nSpecFeatures = len(mzList)
    xLocation = np.array(f["xLocation"]).astype(int)
    yLocation = np.array(f["yLocation"]).astype(int)
    zLocation = np.full(len(yLocation),id)
    col = max(np.unique(xLocation))
    row = max(np.unique(yLocation))
    im = np.zeros((col,row))
    if id==1:
        Combined_MSI = MSI_train
        XCoord = xLocation
        YCoord = yLocation
        ZCoord = zLocation
    else:
        Combined_MSI = np.concatenate((MSI_train,Combined_MSI), axis=0)
        XCoord = np.concatenate((xLocation,XCoord))
        YCoord = np.concatenate((yLocation,YCoord))
        ZCoord = np.concatenate((zLocation,ZCoord))


# ============ KFold Cross Validation:
from sklearn.model_selection import  KFold
from matplotlib.patches import Patch
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

n_folds = 5
kfold = KFold(n_folds,  shuffle=True) 
fig, ax = plt.subplots() 
ij_Training = []; ij_Testing = []
myHF = h5py.File('CV_Values//cv_Idx.h5', 'w')
for ij, (Test_idx, Train_idx) in enumerate(kfold.split(TissueIDs)):
    print("Training: %s Testing:%s" %(Train_idx, Test_idx))
    ij_Training.append(Train_idx)
    ij_Testing.append(Test_idx)
    myHF.create_dataset('indx_Training'+str(ij), data=Train_idx)
    myHF.create_dataset('indx_Testing'+str(ij), data=Test_idx)   
    
    indices = np.array([np.nan] * len(TissueIDs))
    indices[Train_idx] = 1
    indices[Test_idx] = 0
    # Visalize Corss Validation Behavior:
    ax.scatter(range(1,len(indices)+1), [ij + 1] * len(indices),
               c=indices, marker='_', lw=10, cmap=cmap_cv, vmin=-0.2, vmax=1.2)
    yticklabels = list(range(1,n_folds+1))
    ax.set(yticks=np.arange(1,n_folds+1) , yticklabels=yticklabels,
           xlabel='2D MSI Sample index', ylabel="Iteration",
           ylim=[n_folds+1.2,-.2], xlim=[-2, len(TissueIDs)+4])
    ax.set_title('KFold', fontsize=15)
ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.1))],
['Training set', 'Testing set'], loc=(1.02, .8))
myHF.close()

# -------- A function to get raining Data:
def Get_cv_MSI(Combined_MSI,XCoord,YCoord,ZCoord,zSections,CV_idx):
    ij_r=[]; ij_t=[]; MSI_train=[]; MSI_Test=[]
    # Get Training Data:
    for jr,zr in enumerate(CV_idx):        
        if jr==0:
            ij_r = np.argwhere(ZCoord == zSections[zr])
        else:
            ij_r = np.concatenate((ij_r, np.argwhere(ZCoord == zSections[zr])),axis=0)
    MSI_Data = np.squeeze(Combined_MSI[ij_r,])
    XCoord_cv = XCoord[ij_r]
    YCoord_cv = YCoord[ij_r]
    ZCoord_cv = ZCoord[ij_r]

    return MSI_Data,XCoord_cv,YCoord_cv,ZCoord_cv

# ------------------- Train and Test with CV:
zSections = np.unique(ZCoord)
directory = Cd+'/Results_CV'         
meanSpec_Orig_AllData = np.mean(Combined_MSI,axis=0)

# --------- Train model status:
TrainStatus =  int(input("Would you like to Train Your Model? Yes=1; No=0 ... :"))
if TrainStatus== 0:
    print('No Training')
else:
    print('Model Training ...>>>......')
    
hf_cv = h5py.File('CV_Values/cv_Idx.h5', 'r')       
for i in range(n_folds):
    Train_idx = hf_cv['indx_Training'+str(i)][:]
    #Test_idx = hf_cv['indx_Testing'+str(i)][:]
    MSI_train,XCoord_cv,YCoord_cv,ZCoord_cv = Get_cv_MSI(Combined_MSI,XCoord,YCoord,ZCoord,zSections,Train_idx)
	#MSI_Test,XCoord_cv,YCoord_cv,ZCoord_cv = Get_cv_MSI(Combined_MSI,XCoord,YCoord,ZCoord,zSections,Test_idx)
    myzSections = np.unique(ZCoord_cv)     
# ************************* Training **************************************    
    # 1. ====== Initialize the model:
    from Computational_Model import *
    input_shape = (nSpecFeatures, )
    intermediate_dim = 512
    latent_dim = 5
    VAE_BN_Model = VAE_BN(nSpecFeatures,  intermediate_dim, latent_dim)
    myModel, encoder = VAE_BN_Model.get_architecture()
    myModel.summary()  
    # 2. ====== Train the model:
    if TrainStatus==1:
        start_time = time.time()
        history = myModel.fit(MSI_train, epochs=100, batch_size=128, shuffle="batch")   
        myModel.save_weights(directory+'//'+'TrainedModel_'+str(i)+'.h5') 
    else:
        myModel.load_weights(directory+'//'+'TrainedModel_'+str(i)+'.h5');
    # 3. ============= Model Predictions:
    encoded_imgs = encoder.predict(MSI_train) # Learned non-linear manifold
    decoded_imgs = myModel.predict(MSI_train) # Reconstructed Data
    dec_TIC = np.sum(decoded_imgs, axis=-1)
    Latent_mean, Latent_var, Latent_z = encoded_imgs
    
    get_EncFeatures(Latent_z,Train_idx,ZCoord_cv,XCoord_cv,YCoord_cv,directory,i)    
    
    # 4. ============= Plot Average Spectrum:
    mse = mean_squared_error(MSI_train,decoded_imgs)
    meanSpec_Rec = np.mean(decoded_imgs,axis=0) 
    print('mean squared error(mse)  = ', mse)
    meanSpec_Orig = np.mean(MSI_train,axis=0) # TIC-norm original MSI Data
    N_DecImg = decoded_imgs/dec_TIC[:,None]  # TIC-norm reconstructed MSI  Data
    meanSpec_RecTIC = np.mean(N_DecImg,axis=0)
    
    fig, ax = plt.subplots() 
    plt.plot(history.history['loss'])
    plt.ylabel('loss'); plt.xlabel('epoch')
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.savefig(directory+'/'+'Convergence_TrainedModel_'+str(i)+'.tif')
    
    fig, ax = plt.subplots() 
    #plt.figure(figsize=(10, 3))
    plt.plot(mzList,meanSpec_Orig,color = [0, 1, 0,1]); plt.plot(mzList,meanSpec_RecTIC,color = [1, 0, 0,0.6]); 
    plt.savefig(directory+'/'+'Overlay_Training_'+str(i)+'mse_'+str(mse)+'.tif')
    
    # 5. ============== Learn Peaks:
    from LearnPeaks import *
    W_enc = encoder.get_weights()
    # Normalize Weights by multiplying it with std of original data variables
    std_spectra = np.std(MSI_train, axis=0) 
    Beta = 2.5
    Learned_mzBins, Learned_mzPeaks, mzBin_Indx, Real_PeakIdx = LearnPeaks(mzList, W_enc,std_spectra,latent_dim,Beta,meanSpec_Orig_AllData)
    xls_writer = pd.ExcelWriter(directory+'/'+'Peaks_Training_'+str(i)+'.xlsx', engine='xlsxwriter')   
    df_1 = pd.DataFrame({'mz Bins': Learned_mzBins})
    df_2 = pd.DataFrame({'mz Peaks': Learned_mzPeaks})
    df_1.to_excel(xls_writer, sheet_name='Sheet'+str(i))
    df_2.to_excel(xls_writer, sheet_name='Sheet'+str(i),startcol=3)
    workbook  = xls_writer.book
    worksheet = xls_writer.sheets['Sheet'+str(i)]
    
    # 6. ============== Apply CLustering:
    nClusters = 8
    gmm = GaussianMixture(n_components=nClusters,random_state=0).fit(np.squeeze(Latent_z))
    Labels = gmm.predict(np.squeeze(Latent_z))
    get_gmmImage(Train_idx,Labels,nClusters,ZCoord_cv,XCoord_cv,YCoord_cv,directory,i)
                             
hf_cv.close()

# 7. ========== Correlate Clusters with MSI Data:
cluster_id = 7
Corr_Val, CorrRank_ij,MSI_CleanPeaks =  Correlate_Cluster_MSI(cluster_id,Labels,MSI_train,Real_PeakIdx,ZCoord_cv,XCoord_cv,YCoord_cv)

print('m/z', Learned_mzPeaks[CorrRank_ij[0:5]])
print('corr_Value = ', Corr_Val[CorrRank_ij[0:5]])
plt.plot(Learned_mzPeaks,Corr_Val)

# Visualize Correlated Peak at section z: 
mzID = CorrRank_ij[0];
myzSections = np.unique(ZCoord_cv)     
zr=0                    
ij_r = np.argwhere(ZCoord_cv == myzSections[zr])
indx = ij_r[:,0]
xLoc = XCoord_cv[indx]
yLoc = YCoord_cv[indx]
MSI_2D = np.squeeze(MSI_train[indx,mzID])
im_mz = Image_Distribution(MSI_2D,xLoc,yLoc);
plt.imshow(im_mz);
mz_Peak = Learned_mzPeaks[mzID]
print('m/z Peak = ',mz_Peak)

# ============ Get 3D m/z image =================
mzValue = 13972.1
mzId = np.argmin(np.abs(mzList[:] - mzValue))
Get_3Dmz_nifti(Combined_MSI[:,mzId],mzValue,XCoord,YCoord,ZCoord,directory)

# ============== Load Peak Learned by All training models:
myBeta = 1.5
ALL_Peaks_Train = pd.read_excel(directory+'//Peaks_Learned//Beta_'+str(myBeta)+'//Peaks_AllModels_Trained.xlsx')
ALL_Peaks_Train = np.squeeze(np.asarray(ALL_Peaks_Train))
ALL_Peaks_Train = np.nan_to_num(ALL_Peaks_Train)

My_marker= ['v', '*','d','^','s']
Point_Color = plt.cm.jet(np.linspace(0, 1, ALL_Peaks_Train.shape[1]))
plt.figure(figsize=(25, 5))
plt.plot(mzList,meanSpec_Orig_AllData,linewidth=3,c='black'); 
for ij in range(ALL_Peaks_Train.shape[1]):
    Peaks_Train = ALL_Peaks_Train[:,ij]
    Peaks_Train = Peaks_Train[Peaks_Train !=0]
    Train_Peaks_Loc = [np.argmin(np.abs(mzList[:] - Peaks_Train[idx])) for idx in  range(len(Peaks_Train))]
    Mean_PickedPeakst = np.mean(Combined_MSI[:,Train_Peaks_Loc],axis=0)
    plt.scatter(Peaks_Train,Mean_PickedPeakst,marker=My_marker[ij],c=Point_Color[ij]);


Peaks_Vector = ALL_Peaks_Train.reshape(ALL_Peaks_Train.shape[0]*ALL_Peaks_Train.shape[1])
Peaks_Vector_NoZero = Peaks_Vector[Peaks_Vector !=0]
U_Peaks = np.unique(Peaks_Vector_NoZero)
L = len(np.unique(U_Peaks))
#plt.figure(figsize=(20, 5))
n, bins, patches = plt.hist(x=Peaks_Vector_NoZero, bins=U_Peaks); plt.show()

#=== Scatter Plot Frequency:
from matplotlib.ticker import MaxNLocator
New_n = np.append(n,1)
fig, ax = plt.subplots()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#plt.figure(figsize=(30, 5))
plt.scatter(U_Peaks,New_n,c=New_n)
plt.xlabel("m/z")
plt.ylabel("Frequency")

# ====== Bar Plot: Counts of Peak Frequency
N_Freq = [len(np.argwhere(n==ij)) for ij in np.unique(n)]
xValue = [ij for ij in np.unique(n)]
colors = plt.cm.jet(np.linspace(0, 1, len(np.unique(n))))
plt.bar(xValue, N_Freq,color=colors)
plt.xlabel("Frequency")
plt.ylabel("Count")

