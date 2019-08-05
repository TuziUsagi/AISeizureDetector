#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tensorflow as tf

def upsampling(data, hparams):
  data = tf.reshape(data, (1, hparams['inputLength'], 1))
  data = tf.image.resize_images(data, (1, hparams['inputLength']*hparams['upSamplingFactor']), method = 2)
  data = tf.squeeze(data)
  return data

def extract_fn(data_record, hparams, mode):
  # Extract data
  features = {'data': tf.io.FixedLenFeature([hparams['inputLength']], tf.float32),'label':tf.io.FixedLenFeature([1], tf.string)}
  sample = tf.io.parse_single_example(data_record, features)
  sample['label'] = sample['label'][0]
  if(hparams['upSamplingFactor'] > 1):
    sample['data'] = upsampling(sample['data'], hparams)
  # Augment
  sigLen = hparams['inputLength']*hparams['upSamplingFactor']
  shift_value = int(sigLen/4)
  if (hparams['augment'] and mode == 'Train'):
    noiseVector = tf.random.uniform(shape = (sigLen,), minval = -0.1, maxval = 0.1, seed = hparams['randSeed'])
    sample['data'] = sample['data'] + noiseVector
    dirArrayIdx = random.randint(0, 1023)
    shift_step = random.randint(-shift_value, shift_value)
    sample['data'] = tf.roll(sample['data'], shift_step, -1)*hparams['dirArray'][dirArrayIdx]
    signalMin = tf.reduce_min(sample['data'])
    signalMax = tf.reduce_max(sample['data'])
    sample['data'] = sample['data'] - tf.reduce_mean(sample['data'])
    sample['data'] = 2*(sample['data'] - signalMin)/(signalMax - signalMin) - 1
  # One hot encoding
  label = hparams['labelLUtable'].lookup(sample['label'])
  label_one_hot = tf.one_hot(label, hparams['NCLASSES'])
  # Optional casting for higher precision
  if(hparams['64bFilter']):
    sample['data'] = tf.cast(sample['data'], tf.float64)
  # make the reshape
  sample['data'] = tf.reshape(sample['data'], hparams['inputShape'])
  if mode == 'Predict':
    return sample['data']
  elif mode == 'labelOnly':
    return label_one_hot
  else:
    return sample['data'], (label_one_hot, label_one_hot, label_one_hot)

def readDataset(hparams, mode, batch_size):
  num_epochs = None
  filePathList = []
  if(mode == 'Train'):
    filePathList = hparams['train_data']	
  elif(mode == 'Eval'):
    filePathList = hparams['eval_data']
    num_epochs = 2
  else:
    num_epochs = 1
    filePathList = hparams['eval_data']
  dataset = tf.data.TFRecordDataset(filePathList, compression_type='GZIP', buffer_size=1024*1024*1024, num_parallel_reads = (hparams['numOfCores']<<1))
  dataset = dataset.cache()
  dataset = dataset.map(lambda x: extract_fn(x, hparams, mode), num_parallel_calls=hparams['numOfCores'])
  if (mode == 'Train'):
    dataset = dataset.shuffle(buffer_size = int(hparams['trainSetSize']*2)) # Shuffle before batching!!
    dataset = dataset.repeat(num_epochs).batch(batch_size, True) # Drop reminding samples if a full batch cannot be guaranteed to enable JIT complication for higher performance
  elif(mode == 'Eval'):
    dataset = dataset.repeat(num_epochs).batch(batch_size, True)
  else:
    dataset = dataset.batch(batch_size, False)
  dataset = dataset.prefetch(buffer_size = 1)# One batch per step
  return dataset
