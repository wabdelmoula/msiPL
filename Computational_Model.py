# -*- coding: utf-8 -*-
"""
Implementation of msiPL (Abdelmoula et al): Neural Network Architecture (VAE_BN)

    Keras-based implementation of a fully connected variational autoecnoder
    equipped with Batch normalization to correct for covariate shift and improve learning stability

"""

import numpy as np
from keras.layers import Lambda, Input, Dense, ReLU, BatchNormalization
from keras.models import Model
from keras.losses import  categorical_crossentropy
from keras.utils import plot_model
from keras import backend as K


class VAE_BN(object):
    
    def __init__ (self, nSpecFeatures,  intermediate_dim, latent_dim):
        self.nSpecFeatures = nSpecFeatures
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        
    def sampling(self, args):
        """
        Reparameterization trick by sampling from a continuous function (Gaussian with an auxiliary variable ~N(0,1)).
        [see Our methods and for more details see arXiv:1312.6114]
        """
        self.z_mean, self.z_log_var = args
        self.batch = K.shape(self.z_mean)[0]
        self.dim = K.int_shape(self.z_mean)[1]
        self.epsilon = K.random_normal(shape=(self.batch, self.dim)) # random_normal (mean=0 and std=1)
        return self.z_mean + K.exp(0.5 * self.z_log_var) * self.epsilon
    

    def get_architecture(self):
        # =========== 1. Encoder Model================
        input_shape = (self.nSpecFeatures, )
        inputs = Input(shape=input_shape, name='encoder_input')
        h = Dense(self.intermediate_dim)(inputs)
        h = BatchNormalization()(h)
        h = ReLU()(h)
        z_mean = Dense(self.latent_dim, name = 'z_mean')(h)
        z_mean = BatchNormalization()(z_mean)
        z_log_var = Dense(self.latent_dim, name = 'z_log_var')(h)
        z_log_var = BatchNormalization()(z_log_var)
        
        # Reparametrization Tric:
        z = Lambda(self.sampling, output_shape = (self.latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name = 'encoder')
        print("==== Encoder Architecture...")
        encoder.summary()
        # plot_model(encoder, to_file='VAE_BN_encoder.png', show_shapes=True)
        
        # =========== 2. Encoder Model================
        latent_inputs = Input(shape = (self.latent_dim,), name='Latent_Space')
        hdec = Dense(self.intermediate_dim)(latent_inputs)
        hdec = BatchNormalization()(hdec)
        hdec = ReLU()(hdec)
        outputs = Dense(self.nSpecFeatures, activation = 'sigmoid')(hdec)
        decoder = Model(latent_inputs, outputs, name = 'decoder')
        print("==== Decoder Architecture...")
        decoder.summary()       
        # plot_model(decoder, to_file='VAE_BN__decoder.png', show_shapes=True)
        
        #=========== VAE_BN: Encoder_Decoder ================
        outputs = decoder(encoder(inputs)[2])
        VAE_BN_model = Model(inputs, outputs, name='VAE_BN')
        
        # ====== Cost Function (Variational Lower Bound)  ==============
        "KL-div (regularizes encoder) and reconstruction loss (of the decoder): see equation(3) in our paper"
        # 1. KL-Divergence:
        kl_Loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_Loss = K.sum(kl_Loss, axis=-1)
        kl_Loss *= -0.5
        # 2. Reconstruction Loss
        reconstruction_loss = categorical_crossentropy(inputs,outputs) # Use sigmoid at output layer
        reconstruction_loss *= self.nSpecFeatures
        
        # ========== Compile VAE_BN model ===========
        model_Loss = K.mean(reconstruction_loss + kl_Loss)
        VAE_BN_model.add_loss(model_Loss)
        VAE_BN_model.compile(optimizer='adam')
        return VAE_BN_model, encoder






