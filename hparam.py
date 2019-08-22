#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import scipy.signal as signal
from tensorflow.python.compiler.tensorrt import trt_convert

def makeFIlterBank(numFilter, freqStep, filterTap, sampleFreq):
  kernelList = np.zeros([filterTap, 1, numFilter]) # filterLength, inputChannel, OutputChannel
  count = 0
  freqIndex = 0
  while(count < numFilter):
    lowFreqCut = 1.5+freqIndex*freqStep
    highFreqCut = 1.5+(freqIndex+1)*freqStep
    freqIndex = freqIndex + 1
    if(lowFreqCut > 55 and highFreqCut < 65):
      continue
    cutoffList = [lowFreqCut/(sampleFreq/2), highFreqCut/(sampleFreq/2)]
    coef = signal.firwin(filterTap, cutoffList, pass_zero = 'bandpass', scale=True)
    kernelList[:,0,count] = coef
    count = count + 1
  return kernelList

def makeLabelLookupTable(LIST_OF_LABELS, LIST_OF_LABELS_VALUE):
  lookupTableKeys = tf.constant(LIST_OF_LABELS, dtype = tf.string)
  lookupTableVals = tf.constant(LIST_OF_LABELS_VALUE, dtype = tf.int32)
  lookupTableDefaultVals = tf.constant(0, dtype = tf.int32)
  lookupTableInit = tf.lookup.KeyValueTensorInitializer(lookupTableKeys, lookupTableVals, key_dtype = tf.string, value_dtype = tf.int32)
  return tf.lookup.StaticHashTable(lookupTableInit, lookupTableDefaultVals)

def getStepPerEpoch(numOfUnqiueDataPoint, batch_size):
  return int(numOfUnqiueDataPoint / batch_size + 1)

def getInputShape(hparams):
  if(hparams['data_format'] == 'channels_last'):
    return(hparams['inputLength']*hparams['upSamplingFactor'], hparams['inputChannel'])
  else:
    return(hparams['inputChannel'], hparams['inputLength']*hparams['upSamplingFactor'])

def getNormAxis(hparams):
  axis = -2  # norm along channel dimension
  if(hparams['data_format'] == 'channels_last'):
    axis = -1
  else:
    axis = -2
  return axis

def getReduceAxis(hparams):
  if(hparams['data_format'] == 'channels_last'):
    axis = -2
  else:
    axis = -1
  return axis

def gethparam():
  numFilter = 7 # number of filters in the filter bank
  freqStep = 64/numFilter # bandwidth for each filter in the filter bank
  filterTap = 128
  sampleFreq = 250
  LIST_OF_LABELS = ['seizure','baseline']
  LIST_OF_LABELS_VALUE = [0, 1]
  lookupTable = makeLabelLookupTable(LIST_OF_LABELS, LIST_OF_LABELS_VALUE)
  savePrefix = './'
  datasetBucket = 'gs://'
  randomDirArray = np.random.choice([-1.0, 1.0], 1024)
  hparams = {'train_data':[datasetBucket+'train.tfrecord'],
             'eval_data':[datasetBucket+'eval.tfrecord'],
             'logDir': savePrefix+'log/',
             'checkPointDir': savePrefix+'CheckPoint/etebestModel',
             'modelDir': savePrefix+'SavedModel',
             'TPUname': 'v3preemp-central1a2',
             'numOfCores': 4,
              # Input data setup
             'upSamplingFactor': 1,
             '64bFilter': False,
             'trainSetSize': 37264,
             'evalSetSize': 9498,
             'inputLength': 1024,
             'inputChannel': 1,
             'data_format': 'channels_last',
             'batch_size': 4096,
             'augment': True,
              # Network setup
             'dropout': 0.4,
             'fullyConnUseBias': True,
             'kernel1':(64), 'numFilter1':16, 'strides1': 1, 'act1': 'linear', 'use_bias1': True, 'poolsize1': (2), 'poolstride1': (2),
             'kernel2':(32), 'numFilter2':32, 'strides2': 1, 'act2': 'linear', 'use_bias2': True, 'poolsize2': (2), 'poolstride2': (2),
             'kernel3':(16), 'numFilter3':8,  'strides3': 1, 'act3': 'linear', 'use_bias3': True, 'poolsize3': (2), 'poolstride3': (2), 
             'denseLayerSize1': 16,
             'denseLayerSize2': 16,
              # Training setup
             'resume': False,
             'mixedPrecision': False,
             'validation_freq': 1, # How many training epochs finished before each eval event
             'epochs': 36000,  # Total epochs in training loop
             'learning_rate':0.01, 
             'reg_constant': 0.01,
             'AdamBeta1': 0.9,
             'AdamBeta2': 0.999,
             'epsilon': 1e-7,
             # Will calculated automatically, all the values below are dummy values.
             'dirArray': randomDirArray,
             'steps_per_epoch': 1,
             'filterBankSize': (filterTap,1,numFilter),
             'NCLASSES': len(LIST_OF_LABELS),
             'validation_steps': 1,
             'filterKernel': None,
             'GPUmemBytes': 16*1024*1024, #V100: 16GB
             'TRTprecision': 'INT8', #options: ...FP16, ...INT8
             'labelLUtable': lookupTable,
             'randSeed': 1231293,
             'numFilter': numFilter,
             'filterTap': filterTap,
             'inputShape': [1,1],
             'normDim': -1,
             'reduceDim': -1,
             'inputDimWithBatch': 3,
             'monitorVal': 'val_loss'}
  kernelList = makeFIlterBank(numFilter, freqStep, filterTap, int(sampleFreq*hparams['upSamplingFactor']))
  if(hparams['64bFilter']):
    hparams['filterKernel'] = kernelList.astype('float64')
  else:
    hparams['filterKernel'] = kernelList.astype('float32')  
  hparams['steps_per_epoch'] = 8*getStepPerEpoch(hparams['trainSetSize'], hparams['batch_size'])
  hparams['validation_steps'] = getStepPerEpoch(hparams['evalSetSize'], hparams['batch_size'])
  hparams['randSeed'] = int(time.time())
  hparams['inputShape'] = getInputShape(hparams)
  hparams['normDim'] = getNormAxis(hparams) 
  hparams['reduceDim'] = getReduceAxis(hparams)
  hparams['inputDimWithBatch'] = len(hparams['inputShape']) + 1
  return hparams
