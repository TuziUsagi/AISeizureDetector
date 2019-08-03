#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Lambda
import secrets
import string
import numpy as np
# Define model

def bswish(x, mixPrecision):
  return x * tf.keras.activations.sigmoid(x)

def bswishAct(inputLayer, mixPrecision):
  return Lambda(lambda x: bswish(x, mixPrecision))(inputLayer)

def eeg_rnn_tpu(hparams, mode, batch_size = None):
  filterKernelInit = tf.keras.initializers.Constant(hparams['filterKernel'].astype('float32'))  
  l2_regularizer = tf.keras.regularizers.l2(hparams['reg_constant'])
  kernalInitializer = tf.compat.v1.keras.initializers.he_normal(seed = hparams['randSeed'])
  biasInitializer = tf.keras.initializers.zeros()
  normAxis = hparams['normDim']

  InputLayer = tf.keras.Input(shape = hparams['inputShape'], batch_size = batch_size)
  if(hparams['upSamplingFactor'] == 1):
    inputFilterLayer = tf.keras.layers.Conv1D(hparams['numFilter'], 
                                      hparams['filterTap'], 
                                      1,
                                      padding="SAME",
                                      data_format = hparams['data_format'],
                                      activation = 'linear',
                                      use_bias = False,
                                      kernel_initializer = filterKernelInit,
                                      bias_initializer = None,
                                      trainable=False)(InputLayer)
    nonFilteredLSTM = tf.keras.layers.LSTM(units = 1, return_sequences = True)(InputLayer)
  else:
    upsampledLayer = tf.keras.layers.UpSampling1D(size = hparams['upSamplingFactor'])(InputLayer)
    deconvLayer =  tf.keras.layers.Conv1D(1, 
                                      hparams['filterTap'], 
                                      1,
                                      padding="SAME",
                                      data_format = hparams['data_format'],
                                      activation = 'linear',
                                      use_bias = False,
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = None)(upsampledLayer)
    nonFilteredLSTM = tf.keras.layers.LSTM(units = 1, return_sequences = True)(deconvLayer)
    inputFilterLayer = tf.keras.layers.Conv1D(hparams['numFilter'], 
                                      hparams['filterTap'], 
                                      1,
                                      padding="SAME",
                                      data_format = hparams['data_format'],
                                      activation = 'linear',
                                      use_bias = False,
                                      kernel_initializer = filterKernelInit,
                                      bias_initializer = None,
                                      trainable=False)(deconvLayer)								  
  batchNormLayer0 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(inputFilterLayer, False))

  filteredLSTM = tf.keras.layers.LSTM(units = hparams['numFilter'], return_sequences = True)(batchNormLayer0)
  filteredLSTMflatten = tf.keras.layers.Flatten()(filteredLSTM)
  nonFilteredLSTMflatten = tf.keras.layers.Flatten()(nonFilteredLSTM)

  concatLayer = tf.keras.layers.concatenate([filteredLSTMflatten, nonFilteredLSTMflatten])
  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize1'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'mainFullyConn')(concatLayer)
  batchNormLayer1 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(fullConn1, False))
  finalOutLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'softmax',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'mainOutput')(batchNormLayer1)

  EEGmodel = tf.keras.models.Model(inputs = InputLayer, outputs = finalOutLayer)
  return EEGmodel
  
def eeg_rnn_gpu(hparams, mode, batch_size = None):
  filterPolicy = tf.keras.mixed_precision.experimental.Policy('infer')
  if(hparams['64bFilter']): # Apply filterbank in float64 (doube precision)
    filterKernelInit = tf.keras.initializers.Constant(hparams['filterKernel'].astype('float64'), dtype = tf.float64)
    InputLayer = tf.keras.Input(shape = hparams['inputShape'], dtype = tf.float64, batch_size = batch_size)	
  else: # Apply filterbank in float32
    filterKernelInit = tf.keras.initializers.Constant(hparams['filterKernel'].astype('float32'), dtype = tf.float32)
    InputLayer = tf.keras.Input(shape = hparams['inputShape'], dtype = tf.float32, batch_size = batch_size)	

  l2_regularizer = tf.keras.regularizers.l2(hparams['reg_constant'])
  kernalInitializer = tf.compat.v1.keras.initializers.he_normal(seed = hparams['randSeed'])
  biasInitializer = tf.keras.initializers.zeros()
  normAxis = hparams['normDim']

  if(hparams['upSamplingFactor'] == 1):
    inputFilterLayer = tf.keras.layers.Conv1D(hparams['numFilter'], 
                                      hparams['filterTap'], 
                                      1,
                                      padding="SAME",
                                      data_format = hparams['data_format'],
                                      activation = 'linear',
                                      use_bias = False,
                                      kernel_initializer = filterKernelInit,
                                      dtype = filterPolicy,
                                      bias_initializer = None,
                                      trainable=False)(InputLayer)
    if(hparams['mixedPrecision']): # Convert to target data type after filterbank (if no upsampling is needed)
      castedInputLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float16))(InputLayer)
    elif(hparams['64bFilter']):
      castedInputLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(InputLayer)
    else:
      castedInputLayer = InputLayer
    nonFilteredLSTM = tf.keras.layers.CuDNNLSTM(units = 1, return_sequences = True)(castedInputLayer)
  else:
    upsampledLayer = tf.keras.layers.UpSampling1D(size = hparams['upSamplingFactor'], dtype = filterPolicy)(InputLayer)
    deconvLayer =  tf.keras.layers.Conv1D(1, 
                                      hparams['filterTap'], 
                                      1,
                                      padding="SAME",
                                      data_format = hparams['data_format'],
                                      activation = 'linear',
                                      use_bias = False,
                                      kernel_initializer = kernalInitializer,
                                      dtype = filterPolicy,
                                      bias_initializer = None)(upsampledLayer)
    if(hparams['mixedPrecision']): # Convert input to target data type after filterbank (after upsampling)
      castedDeconvLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float16))(deconvLayer)
    elif(hparams['64bFilter']):
      castedDeconvLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(deconvLayer)
    else:
      castedDeconvLayer = deconvLayer
    nonFilteredLSTM = tf.keras.layers.CuDNNLSTM(units = 1, return_sequences = True)(castedDeconvLayer)
    inputFilterLayer = tf.keras.layers.Conv1D(hparams['numFilter'], 
                                      hparams['filterTap'], 
                                      1,
                                      padding="SAME",
                                      data_format = hparams['data_format'],
                                      activation = 'linear',
                                      use_bias = False,
                                      kernel_initializer = filterKernelInit,
                                      dtype = filterPolicy,
                                      bias_initializer = None,
                                      trainable=False)(deconvLayer)								  
  batchNormLayer0 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(inputFilterLayer, hparams['mixedPrecision'])) # Use default values
  if(hparams['mixedPrecision']):
    castedFilteredInput = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float16))(batchNormLayer0)
  elif(hparams['64bFilter']):
    castedFilteredInput = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(batchNormLayer0)
  else:
    castedFilteredInput = batchNormLayer0

  filteredLSTM = tf.keras.layers.CuDNNLSTM(units = hparams['numFilter'], return_sequences = True)(castedFilteredInput)
  filteredLSTMflatten = tf.keras.layers.Flatten()(filteredLSTM)
  nonFilteredLSTMflatten = tf.keras.layers.Flatten()(nonFilteredLSTM)

  concatLayer = tf.keras.layers.concatenate([filteredLSTMflatten, nonFilteredLSTMflatten])

  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize1'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'mainFullyConn')(concatLayer)
  batchNormLayer1 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(fullConn1, hparams['mixedPrecision']))
  finalLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'mainOutput')(batchNormLayer1)

  if(hparams['mixedPrecision']):
    castedFinalLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(finalLayer)
    finalOutLayer = tf.keras.layers.Activation('softmax', name = 'mainOutMix')(castedFinalLayer)
  else:
    finalOutLayer = tf.keras.layers.Activation('softmax', name = 'mainOut')(finalLayer) 

  EEGmodel = tf.keras.models.Model(inputs = InputLayer, outputs = finalOutLayer)

  return EEGmodel

