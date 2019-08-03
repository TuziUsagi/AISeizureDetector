#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Lambda
import secrets
import string
# Define model

def bswish(x, mixPrecision):
  return x * tf.keras.activations.sigmoid(x)

def bswishAct(inputLayer, mixPrecision):
  return Lambda(lambda x: bswish(x, mixPrecision))(inputLayer)

def eeg_inception_unit_maxpool_largefilter(inputLayer, hparams, mode):
  l2_regularizer = tf.keras.regularizers.l2(hparams['reg_constant'])
  kernalInitializer = tf.compat.v1.keras.initializers.he_normal(seed = hparams['randSeed'])
  biasInitializer = tf.keras.initializers.zeros()
  normAxis = hparams['normDim']

  convLayer1 = tf.keras.layers.Conv1D(hparams['numFilter1'], 
                                      hparams['kernel1'], 
                                      hparams['strides1'],
                                      dilation_rate = hparams['upSamplingFactor']*2,
                                      padding="same",
                                      data_format = hparams['data_format'],
                                      activation = hparams['act1'],
                                      use_bias = hparams['use_bias1'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(inputLayer)

  batchNormLayer1 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(convLayer1, hparams['mixedPrecision']))
  poolLayer1 = tf.keras.layers.MaxPool1D(pool_size = hparams['poolsize1'],
                                         strides = hparams['poolstride1'],
                                         padding = "same",
                                         trainable = False,
                                         data_format = hparams['data_format'])(batchNormLayer1) # Output: 512, 96


  convLayer2 = tf.keras.layers.Conv1D(hparams['numFilter2'], 
                                      hparams['kernel2'], 
                                      hparams['strides2'],
                                      dilation_rate = hparams['upSamplingFactor']*2,
                                      padding="same",
                                      data_format = hparams['data_format'],
                                      activation = hparams['act2'],
                                      use_bias = hparams['use_bias2'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(poolLayer1)
  batchNormLayer2 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(convLayer2, hparams['mixedPrecision']))
  poolLayer2 = tf.keras.layers.MaxPool1D(pool_size = hparams['poolsize2'],
                                         strides = hparams['poolstride2'],
                                         padding = "same",
                                         trainable = False,
                                         data_format = hparams['data_format'])(batchNormLayer2) # Output: 256, 128

  flatten = tf.keras.layers.Flatten()(poolLayer2)
  
  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize1'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer)(flatten)
  batchNormLayer3 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(fullConn1, hparams['mixedPrecision']))
  if(mode == 'Train'): # Only apply dropout during training
    finalInput = tf.keras.layers.Dropout(hparams['dropout'], name = 'MLfilterAUX_drop2')(batchNormLayer3)
  else:
    finalInput = batchNormLayer3
  auxFinalLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer)(finalInput)
  if(hparams['mixedPrecision']):
    castedFinalLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(auxFinalLayer)
    auxOutLayer = tf.keras.layers.Activation('softmax', name = 'MLfilterAUX')(castedFinalLayer)
  else:
    auxOutLayer = tf.keras.layers.Activation('softmax', name = 'MLfilterAUX')(auxFinalLayer) 
  return batchNormLayer3, auxOutLayer
  
def eeg_inception_unit_avgpool_largefilter(inputLayer, hparams, mode):
  l2_regularizer = tf.keras.regularizers.l2(hparams['reg_constant'])
  kernalInitializer = tf.compat.v1.keras.initializers.he_normal(seed = hparams['randSeed'])
  biasInitializer = tf.keras.initializers.zeros()
  normAxis = hparams['normDim']
  convLayer1 = tf.keras.layers.Conv1D(hparams['numFilter1'], 
                                      hparams['kernel1'], 
                                      hparams['strides1'],
                                      dilation_rate = hparams['upSamplingFactor']*2,
                                      padding="same",
                                      data_format = hparams['data_format'],
                                      activation = hparams['act1'],
                                      use_bias = hparams['use_bias1'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(inputLayer)
  batchNormLayer1 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(convLayer1, hparams['mixedPrecision']))
  poolLayer1 = tf.keras.layers.AveragePooling1D(pool_size = hparams['poolsize1'],
                                         strides = hparams['poolstride1'],
                                         padding = "same",
                                         trainable = False,
                                         data_format = hparams['data_format'])(batchNormLayer1) # Output: 512, 96
 
  convLayer2 = tf.keras.layers.Conv1D(hparams['numFilter2'], 
                                      hparams['kernel2'], 
                                      hparams['strides2'],
                                      dilation_rate = hparams['upSamplingFactor']*2,
                                      padding="same",
                                      data_format = hparams['data_format'],
                                      activation = hparams['act2'],
                                      use_bias = hparams['use_bias2'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(poolLayer1)
  batchNormLayer2 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(convLayer2, hparams['mixedPrecision']))
  poolLayer2 = tf.keras.layers.AveragePooling1D(pool_size = hparams['poolsize2'],
                                         strides = hparams['poolstride2'],
                                         padding = "same",
                                         trainable = False,
                                         data_format = hparams['data_format'])(batchNormLayer2) # Output: 256, 128

  flatten = tf.keras.layers.Flatten()(poolLayer2)
  
  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize1'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer)(flatten)
  batchNormLayer3 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(fullConn1, hparams['mixedPrecision']))
  if(mode == 'Train'): # Only apply dropout during training
    finalInput = tf.keras.layers.Dropout(hparams['dropout'], name = 'ALfilterAUX_drop2')(batchNormLayer3)
  else:
    finalInput = batchNormLayer3

  auxFinalLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer)(finalInput)
  if(hparams['mixedPrecision']):
    castedFinalLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(auxFinalLayer)
    auxOutLayer = tf.keras.layers.Activation('softmax', name = 'ALfilterAUX')(castedFinalLayer)
  else:
    auxOutLayer = tf.keras.layers.Activation('softmax', name = 'ALfilterAUX')(auxFinalLayer) 
  return batchNormLayer3, auxOutLayer

  
def eeg_inception_unit_maxpool_smallfilter(inputLayer, hparams, mode):
  l2_regularizer = tf.keras.regularizers.l2(hparams['reg_constant'])
  kernalInitializer = tf.compat.v1.keras.initializers.he_normal(seed = hparams['randSeed'])
  biasInitializer = tf.keras.initializers.zeros()
  normAxis = hparams['normDim']
  convLayer1 = tf.keras.layers.Conv1D(hparams['numFilter1'], 
                                      hparams['kernel1'], 
                                      hparams['strides1'],
                                      padding="same",
                                      data_format = hparams['data_format'],
                                      activation = hparams['act1'],
                                      use_bias = hparams['use_bias1'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(inputLayer)  # Output: 2048, 96

  batchNormLayer1 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(convLayer1, hparams['mixedPrecision']))
  poolLayer1 = tf.keras.layers.MaxPool1D(pool_size = hparams['poolsize1'],
                                         strides = hparams['poolstride1'],
                                         padding = "same",
                                         trainable = False,
                                         data_format = hparams['data_format'])(batchNormLayer1) # Output: 1024, 96
 
  convLayer2 = tf.keras.layers.Conv1D(hparams['numFilter2'], 
                                      hparams['kernel2'], 
                                      hparams['strides2'],
                                      padding="same",
                                      data_format = hparams['data_format'],
                                      activation = hparams['act2'],
                                      use_bias = hparams['use_bias2'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(poolLayer1)
  batchNormLayer2 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(convLayer2, hparams['mixedPrecision']))  # Output: 512, 128
  poolLayer2 = tf.keras.layers.MaxPool1D(pool_size = hparams['poolsize2'],
                                         strides = hparams['poolstride2'],
                                         padding = "same",
                                         trainable = False,
                                         data_format = hparams['data_format'])(batchNormLayer2) # Output: 128, 128

  flatten = tf.keras.layers.Flatten()(poolLayer2)
  
  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize1'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer)(flatten)
  batchNormLayer3 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(fullConn1, hparams['mixedPrecision']))
  if(mode == 'Train'): # Only apply dropout during training
    finalInput = tf.keras.layers.Dropout(hparams['dropout'], name = 'MSfilterAUX_drop2')(batchNormLayer3)
  else:
    finalInput = batchNormLayer3
  auxFinalLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer)(finalInput)
  if(hparams['mixedPrecision']):
    castedFinalLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(auxFinalLayer)
    auxOutLayer = tf.keras.layers.Activation('softmax', name = 'MSfilterAUX')(castedFinalLayer)
  else:
    auxOutLayer = tf.keras.layers.Activation('softmax', name = 'MSfilterAUX')(auxFinalLayer) 
  return batchNormLayer3, auxOutLayer
  
def eeg_inception_unit_avgpool_smallfilter(inputLayer, hparams, mode):
  l2_regularizer = tf.keras.regularizers.l2(hparams['reg_constant'])
  kernalInitializer = tf.compat.v1.keras.initializers.he_normal(seed = hparams['randSeed'])
  biasInitializer = tf.keras.initializers.zeros()
  normAxis = hparams['normDim']
  convLayer1 = tf.keras.layers.Conv1D(hparams['numFilter1'], 
                                      hparams['kernel1'], 
                                      hparams['strides1'],
                                      padding="same",
                                      data_format = hparams['data_format'],
                                      activation = hparams['act1'],
                                      use_bias = hparams['use_bias1'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(inputLayer)

  batchNormLayer1 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(convLayer1, hparams['mixedPrecision']))  
  poolLayer1 = tf.keras.layers.AveragePooling1D(pool_size = hparams['poolsize1'],
                                         strides = hparams['poolstride1'],
                                         padding = "same",
                                         trainable = False,
                                         data_format = hparams['data_format'])(batchNormLayer1) # Output: 512, 96
  
  convLayer2 = tf.keras.layers.Conv1D(hparams['numFilter2'], 
                                      hparams['kernel2'], 
                                      hparams['strides2'],
                                      padding="same",
                                      data_format = hparams['data_format'],
                                      activation = hparams['act2'],
                                      use_bias = hparams['use_bias2'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(poolLayer1)
  batchNormLayer2 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(convLayer2, hparams['mixedPrecision']))
  poolLayer2 = tf.keras.layers.AveragePooling1D(pool_size = hparams['poolsize2'],
                                         strides = hparams['poolstride2'],
                                         padding = "same",
                                         data_format = hparams['data_format'])(batchNormLayer2) # Output: 256, 128
								

  flatten = tf.keras.layers.Flatten()(poolLayer2)
  
  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize1'],
                                    activation = 'elu',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer)(flatten)
  batchNormLayer3 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(fullConn1, hparams['mixedPrecision']))

  if(mode == 'Train'): # Only apply dropout during training
    finalInput = tf.keras.layers.Dropout(hparams['dropout'], name = 'ASfilterAUX_drop2')(batchNormLayer3)
  else:
    finalInput = batchNormLayer3
  auxFinalLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer)(finalInput)
  if(hparams['mixedPrecision']):
    castedFinalLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(auxFinalLayer)
    auxOutLayer = tf.keras.layers.Activation('softmax', name = 'ASfilterAUX')(castedFinalLayer)
  else:
    auxOutLayer = tf.keras.layers.Activation('softmax', name = 'ASfilterAUX')(auxFinalLayer) 
  return batchNormLayer3, auxOutLayer

def eeg_inception_tpu(hparams, mode, batch_size = None):
  filterKernelInit = tf.keras.initializers.Constant(hparams['filterKernel'].astype('float32'))  
  l2_regularizer = tf.keras.regularizers.l2(hparams['reg_constant'])
  kernalInitializer = tf.compat.v1.keras.initializers.he_normal(seed = hparams['randSeed'])
  biasInitializer = tf.keras.initializers.zeros()
  normAxis = hparams['normDim']

  InputLayer = tf.keras.Input(shape = hparams['inputShape'], batch_size = batch_size)
  
  if(hparams['upSamplingFactor'] < 2):
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
  else:
    upsampledLayer = tf.keras.layers.UpSampling1D(size = hparams['upSamplingFactor'])(InputLayer)
    deconvLayer =  tf.keras.layers.Conv1D(1, 
                                      128, 
                                      1,
                                      padding="SAME",
                                      data_format = hparams['data_format'],
                                      activation = 'linear',
                                      use_bias = False,
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = None)(upsampledLayer)
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
  batchNormLayer0 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(inputFilterLayer, hparams['mixedPrecision'])) # Use default values

  (maxPoolLargeFilter, maxPoolLargeFilterAux) = eeg_inception_unit_maxpool_largefilter(batchNormLayer0, hparams, mode)
  (avgPoolLargeFilter, avgPoolLargeFilterAux) = eeg_inception_unit_avgpool_largefilter(batchNormLayer0, hparams, mode)
  (maxPoolSmallFilter, maxPoolSmallFilterAux) = eeg_inception_unit_maxpool_smallfilter(batchNormLayer0, hparams, mode)
  (avgPoolSmallFilter, avgPoolSmallFilterAux) = eeg_inception_unit_avgpool_smallfilter(batchNormLayer0, hparams, mode)
  concatLayer = tf.keras.layers.concatenate([maxPoolLargeFilter, avgPoolLargeFilter, maxPoolSmallFilter, avgPoolSmallFilter])
  if(mode == 'Train'): # Only apply dropout during training
    fullConnInput = tf.keras.layers.Dropout(hparams['dropout'], name = 'mainDropout')(concatLayer)
  else:
    fullConnInput = concatLayer
  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize2'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'mainFullyConn')(fullConnInput)
  batchNormLayer4 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(fullConn1, hparams['mixedPrecision']))

  finalOutLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'softmax',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'mainOutput')(batchNormLayer4)

  EEGmodel = tf.keras.models.Model(inputs = [InputLayer], outputs = [finalOutLayer, maxPoolLargeFilterAux, avgPoolLargeFilterAux, maxPoolSmallFilterAux, avgPoolSmallFilterAux])
  return EEGmodel
  
def eeg_inception_gpu(hparams, mode, batch_size = None):
  filterPolicy = tf.keras.mixed_precision.experimental.Policy('infer')
  if(hparams['64bFilter']):
    filterKernelInit = tf.keras.initializers.Constant(hparams['filterKernel'].astype('float64'), dtype = tf.float64)
    InputLayer = tf.keras.Input(shape = hparams['inputShape'], dtype = tf.float64, batch_size = batch_size)	
  else:
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
                                      trainable=True)(InputLayer)
  else:
    upsampledLayer = tf.keras.layers.UpSampling1D(size = hparams['upSamplingFactor'], dtype = filterPolicy)(InputLayer)
    deconvLayer =  tf.keras.layers.Conv1D(1, 
                                      128, 
                                      1,
                                      padding="SAME",
                                      data_format = hparams['data_format'],
                                      activation = 'linear',
                                      use_bias = False,
                                      kernel_initializer = kernalInitializer,
                                      dtype = filterPolicy,
                                      bias_initializer = None)(upsampledLayer)
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
                                      trainable=True)(deconvLayer)								  
  batchNormLayer0 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(inputFilterLayer, hparams['mixedPrecision'])) # Use default values
  
  if(hparams['mixedPrecision']):
    castedInput = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float16))(batchNormLayer0)
  elif(hparams['64bFilter']):
    castedInput = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(batchNormLayer0)
  else:
    castedInput = batchNormLayer0
  (maxPoolLargeFilter, maxPoolLargeFilterAux) = eeg_inception_unit_maxpool_largefilter(castedInput, hparams, mode)
  (avgPoolLargeFilter, avgPoolLargeFilterAux) = eeg_inception_unit_avgpool_largefilter(castedInput, hparams, mode)
  (maxPoolSmallFilter, maxPoolSmallFilterAux) = eeg_inception_unit_maxpool_smallfilter(castedInput, hparams, mode)
  (avgPoolSmallFilter, avgPoolSmallFilterAux) = eeg_inception_unit_avgpool_smallfilter(castedInput, hparams, mode)
  concatLayer = tf.keras.layers.concatenate([maxPoolLargeFilter, avgPoolLargeFilter, maxPoolSmallFilter, avgPoolSmallFilter])

  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize2'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'mainFullyConn')(concatLayer)
  batchNormLayer4 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(fullConn1, hparams['mixedPrecision']))

  if(mode == 'Train'): # Only apply dropout during training
    finalInput = tf.keras.layers.Dropout(hparams['dropout'], name = 'mainDropout')(batchNormLayer4)
  else:
    finalInput = batchNormLayer4

  finalLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'mainOutput')(finalInput)

  if(hparams['mixedPrecision']):
    castedFinalLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(finalLayer)
    finalOutLayer = tf.keras.layers.Activation('softmax', name = 'mainOutMix')(castedFinalLayer)
  else:
    finalOutLayer = tf.keras.layers.Activation('softmax', name = 'mainOut')(finalLayer) 

  EEGmodel = tf.keras.models.Model(inputs = [InputLayer], outputs = [finalOutLayer, maxPoolLargeFilterAux, avgPoolLargeFilterAux, maxPoolSmallFilterAux, avgPoolSmallFilterAux])

  return EEGmodel



