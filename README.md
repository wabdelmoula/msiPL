**msiPL**
---------
Deep Learning based implementation for analysis of mass spectrometry imaging data

This readme file shows how to properly run the msiPL code 

**Paper:** Walid Abdelmoula et al, msiPL: Non-linear Manifold and Peak Learning of Mass Spectrometry Imaging Data Using Artificial Neural Networks, bioRxiv, 2020 

**License:** The msiPL code is shared under the 3D Slicer Software License agreement.

**Installations: Software and Libraries** 
--------

We have implemented our machine learning model using the following software items:

1- Python(3.6.4)

2- Keras (2.1.5-tf) with a Tensorflow(1.8.0) backend.

3- Packages: numpy(1.14.2), sklearn(0.19.1), scipy(1.0.0), and h5py(2.7.1)

4- We implemented this model on Windows 10 PC workstation(Intel Xenon 3.3GHz, 512 GB RAM, 64-bit Windows, 2 GPUs NVIDIA TITAN Xp).
	
 Demo 
 ---------------
 
* How to run the code?

	1- "msiPL_Run.py" is the main file that you should run first. The file should be running in a sequential manner, and we have
	provided required comments for instructions and guidance. In this file you will be able to:
	
		1.1. Load a dataset.
		1.2. Load the computational neural network architecture (VAE_BN).
		1.3. Train the model.
		1.3. Non-linear manifold learning and data visualization (non-linear dimensionality reduction)
		1.4. Evaluate the learning quality by estimation and reconstruction of the original data
		1.5. Peak Learning learning (Equation#4): to get a smaller list of informative peaks.
		1.6. Perform data clustering (GMM).
		1.7. Identify localized peaks within each cluster.
		
	2- "Computational_Model.py": implementation of the fully connected variational autoencoder, and regularized
	    with batch normalization.
	
	3- "LearnPeaks.py": implementation of a function that identifies peaks of interest. 
		It should be called after training the model, as instructed in "msiPL_Run.py".
		
	4. "msiPL_ForTesting.py": ultra-fast analysis on test data without any prior peak picking.
		You will need first to load the trained model from step#1 ("msiPL_Run.py").

If you used this implementation:
------
please cite the paper by Abdelmoula et al, msiPL: https://www.biorxiv.org/content/10.1101/2020.08.13.250142v1.abstract
