#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Lambda
# Define model

def bswish(x, mixPrecision):
  return x * tf.keras.activations.sigmoid(x)

def bswishAct(inputLayer, mixPrecision):
  return Lambda(lambda x: bswish(x, mixPrecision))(inputLayer)

def eeg_inception_unit(inputLayer, hparams, mode, pooling, dilate, name):
  l2_regularizer = tf.keras.regularizers.l2(hparams['reg_constant'])
  kernalInitializer = tf.compat.v1.keras.initializers.he_normal(seed = hparams['randSeed'])
  biasInitializer = tf.keras.initializers.zeros()
  normAxis = hparams['normDim']
  convLayer1 = tf.keras.layers.Conv1D(hparams['numFilter1'], 
                                      hparams['kernel1'], 
                                      hparams['strides1'],
                                      padding="same",
                                      dilation_rate = dilate,
                                      data_format = hparams['data_format'],
                                      activation = hparams['act1'],
                                      use_bias = hparams['use_bias1'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      name = name + '_conv1',
                                      kernel_regularizer = l2_regularizer)(inputLayer)  # Output: 2048, 96
  #batchNormLayer1 = tf.keras.layers.BatchNormalization(normAxis, name = name + '_batchNorm1')(convLayer1)
  batchNormLayer1 = tf.keras.layers.BatchNormalization(normAxis, name = name + '_batchNorm1')(bswishAct(convLayer1, hparams['mixedPrecision']))
  if(pooling == 'Max'):
    poolLayer1 = tf.keras.layers.MaxPool1D(pool_size = hparams['poolsize1'],
                                           strides = hparams['poolstride1'],
                                           padding = "same",
                                           name = name +'_MAXpool1',
                                           data_format = hparams['data_format'])(batchNormLayer1)
  else:
    poolLayer1 = tf.keras.layers.AveragePooling1D(pool_size = hparams['poolsize1'],
                                           strides = hparams['poolstride1'],
                                           padding = "same",
                                           name = name +'_AVGpool1',
                                           data_format = hparams['data_format'])(batchNormLayer1)
 
  convLayer2 = tf.keras.layers.Conv1D(hparams['numFilter2'], 
                                      hparams['kernel2'], 
                                      hparams['strides2'],
                                      padding="same",
                                      dilation_rate = dilate,
                                      data_format = hparams['data_format'],
                                      activation = hparams['act2'],
                                      use_bias = hparams['use_bias2'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      name = name + '_conv2',
                                      kernel_regularizer = l2_regularizer)(poolLayer1)
  #batchNormLayer2 = tf.keras.layers.BatchNormalization(normAxis, name = name + '_batchNorm2')(convLayer2)  # Output: 512, 128
  batchNormLayer2 = tf.keras.layers.BatchNormalization(normAxis, name = name + '_batchNorm2')(bswishAct(convLayer2, hparams['mixedPrecision']))  # Output: 512, 128
  if(pooling == 'Max'):
    poolLayer2 = tf.keras.layers.MaxPool1D(pool_size = hparams['poolsize2'],
                                           strides = hparams['poolstride2'],
                                           padding = "same",
                                           name = name +'_MAXpool2',
                                           data_format = hparams['data_format'])(batchNormLayer2)
  else:
    poolLayer2 = tf.keras.layers.AveragePooling1D(pool_size = hparams['poolsize2'],
                                           strides = hparams['poolstride2'],
                                           padding = "same",
                                           name = name +'_AVGpool2',
                                           data_format = hparams['data_format'])(batchNormLayer2)
  
  convLayer3 = tf.keras.layers.Conv1D(hparams['numFilter3'], 
                                      hparams['kernel3'], 
                                      hparams['strides3'],
                                      padding="same",
                                      dilation_rate = dilate,
                                      name = name + '_conv3',
                                      data_format = hparams['data_format'],
                                      activation = hparams['act3'],
                                      use_bias = hparams['use_bias3'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(poolLayer2)
  #batchNormLayer3 = tf.keras.layers.BatchNormalization(normAxis, name = name + '_batchNorm3')(convLayer3)
  batchNormLayer3 = tf.keras.layers.BatchNormalization(normAxis, name = name + 'batchNorm3')(bswishAct(convLayer3, hparams['mixedPrecision'])) 
  if(pooling == 'Max'):
    poolLayer3 = tf.keras.layers.MaxPool1D(pool_size = hparams['poolsize3'],
                                           strides = hparams['poolstride3'],
                                           padding = "same",
                                           name = name +'_MAXpool3',
                                           data_format = hparams['data_format'])(batchNormLayer3)
  else:
    poolLayer3 = tf.keras.layers.AveragePooling1D(pool_size = hparams['poolsize3'],
                                           strides = hparams['poolstride3'],
                                           padding = "same",
                                           name = name +'_AVGpool3',
                                           data_format = hparams['data_format'])(batchNormLayer3)
			
  flatten = tf.keras.layers.Flatten(name = name + '_flatten')(poolLayer3)
  
  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize1'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    name = name + '_FC1',
                                    kernel_regularizer = l2_regularizer)(flatten)
  #batchNormLayer4 = tf.keras.layers.BatchNormalization(normAxis, name = name + 'batchNorm4')(fullConn1)
  batchNormLayer4 = tf.keras.layers.BatchNormalization(normAxis, name = name + 'batchNorm4')(bswishAct(fullConn1, hparams['mixedPrecision']))
  if(mode == 'Train'): # Only apply dropout during training
    finalInput = tf.keras.layers.Dropout(hparams['dropout'], name = name + '_drop1')(batchNormLayer4)
  else:
    finalInput = batchNormLayer4
  auxFinalLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'linear',
                                    name = name + '_FC2',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer)(finalInput)
  if(hparams['mixedPrecision']):
    castedFinalLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(auxFinalLayer)
    auxOutLayer = tf.keras.layers.Activation('softmax', name = name + '_mixedout')(castedFinalLayer)
  else:
    auxOutLayer = tf.keras.layers.Activation('softmax', name = name + '_out')(auxFinalLayer) 
  return poolLayer2, auxOutLayer

def eeg_inception(hparams, mode, batch_size = None):
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
  
  inputFilterLayer = tf.keras.layers.Conv1D(hparams['numFilter'], 
                                      hparams['filterTap'], 
                                      1,
                                      padding="SAME",
                                      dtype = filterPolicy,									
                                      data_format = hparams['data_format'],
                                      activation = 'linear',
                                      use_bias = False,
                                      trainable = False,
                                      kernel_initializer = filterKernelInit)(InputLayer)
  concatInput = tf.keras.layers.concatenate([InputLayer, inputFilterLayer])
 
  batchNormLayer0 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(concatInput, hparams['mixedPrecision'])) # Use default values
  #batchNormLayer0 = tf.keras.layers.BatchNormalization(normAxis, name = 'mainBatchNorm0')(inputFilterLayer) # Use default values
  if(hparams['mixedPrecision']):
    processedInput = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float16), name = 'inputCast16b')(batchNormLayer0)
  elif(hparams['64bFilter']):
    processedInput = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32), name = 'inputCast32b')(batchNormLayer0)
  else:
    processedInput = batchNormLayer0
  #4 Inception blocks 
  (maxPoolLargeFilter, maxPoolLargeFilterAux) = eeg_inception_unit(processedInput, hparams, mode, 'Max', 4*hparams['upSamplingFactor'], 'Lfilter')
  #(avgPoolLargeFilter, avgPoolLargeFilterAux) = eeg_inception_unit_avgpool_largefilter(processedInput, hparams, mode)
  (maxPoolSmallFilter, maxPoolSmallFilterAux) = eeg_inception_unit(processedInput, hparams, mode, 'Max', hparams['upSamplingFactor'], 'Sfilter')
  #(avgPoolSmallFilter, avgPoolSmallFilterAux) = eeg_inception_unit_avgpool_smallfilter(processedInput, hparams, mode)
  
  concatInceptionLayer = tf.keras.layers.concatenate([maxPoolLargeFilter, maxPoolSmallFilter])
  convLayer3 = tf.keras.layers.Conv1D(hparams['numFilter3'], 
                                      hparams['kernel3'], 
                                      hparams['strides3'],
                                      padding="same",
                                      data_format = hparams['data_format'],
                                      activation = hparams['act3'],
                                      use_bias = hparams['use_bias3'],
                                      kernel_initializer = kernalInitializer,
                                      bias_initializer = biasInitializer,
                                      kernel_regularizer = l2_regularizer)(concatInceptionLayer)
  batchNormLayer3 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(convLayer3, hparams['mixedPrecision']))
  #batchNormLayer3 = tf.keras.layers.BatchNormalization(normAxis)(convLayer3)
  poolLayer3 = tf.keras.layers.MaxPool1D(pool_size = hparams['poolsize3'],
                                         strides = hparams['poolstride3'],
                                         padding = "same",
                                         data_format = hparams['data_format'])(batchNormLayer3) # Output: 256, 128
  mainFlatten = tf.keras.layers.Flatten()(poolLayer3)
  fullConn1 = tf.keras.layers.Dense(hparams['denseLayerSize2'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'mainFullyConn')(mainFlatten)
  #batchNormLayer4 = tf.keras.layers.BatchNormalization(normAxis)(fullConn1)
  batchNormLayer4 = tf.keras.layers.BatchNormalization(normAxis)(bswishAct(fullConn1, hparams['mixedPrecision']))
  
  if(mode == 'Train'): # Only apply dropout during training
    fullConnInput = tf.keras.layers.Dropout(hparams['dropout'], name = 'mainDropout')(batchNormLayer4)
  else:
    fullConnInput = batchNormLayer4
  finalLayer = tf.keras.layers.Dense(hparams['NCLASSES'],
                                    activation = 'linear',
                                    use_bias = hparams['fullyConnUseBias'],
                                    kernel_initializer = kernalInitializer,
                                    bias_initializer = biasInitializer,
                                    kernel_regularizer = l2_regularizer,
                                    name = 'main_FC2')(fullConnInput)
  if(hparams['mixedPrecision']):
    castedFinalLayer = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(finalLayer)
    finalOutLayer = tf.keras.layers.Activation('softmax', name = 'mainOutMix')(castedFinalLayer)
  else:
    finalOutLayer = tf.keras.layers.Activation('softmax', name = 'mainOut')(finalLayer) 
  EEGmodel = tf.keras.models.Model(inputs = [InputLayer], outputs = [finalOutLayer, maxPoolLargeFilterAux, maxPoolSmallFilterAux])
  return EEGmodel
