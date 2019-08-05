#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inception import eeg_inception as create_model
from readDataset import readDataset
from hparam import gethparam
from tensorflow.python import debug as tf_debug
import os

#Setup network
hparams = gethparam()
hparams['64bFilter'] = False
hparams['mixedPrecision'] = False
# Get dataset
trainData = readDataset(hparams, 'Train', hparams['batch_size'])
evalData = readDataset(hparams, 'Eval', hparams['batch_size'])
# Get optimizer
EEGOptimizer = tf.keras.optimizers.Adam(learning_rate = hparams['learning_rate'], beta_1 = hparams['AdamBeta1'], beta_2 = hparams['AdamBeta2'], epsilon = hparams['epsilon'])
 
# Get loss function
lossFunc = tf.keras.losses.CategoricalCrossentropy()

# Metrics
accMetrics = tf.keras.metrics.CategoricalAccuracy()

checkpointWeight_callback = tf.keras.callbacks.ModelCheckpoint(hparams['checkPointDir']+'.h5', save_best_only=True, load_weights_on_restart=False, monitor = hparams['monitorVal'], save_weights_only = False)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(hparams['checkPointDir']+'_restore.h5', save_best_only=False, load_weights_on_restart=hparams['resume'], monitor = hparams['monitorVal'], save_weights_only = False)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, verbose = 1, min_delta = 0.001, cooldown = 3, min_lr=0.0001)
callbacksList = [checkpointWeight_callback, checkpoint_callback, reduce_lr]

# Get TPU:
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = hparams['TPUname'])
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
# Get model
with strategy.scope():
  model = create_model(hparams, 'Train', hparams['batch_size])
  model.summary()
  model.compile(optimizer = EEGOptimizer, loss = lossFunc, metrics = [accMetrics], loss_weights=[1., 0.15, 0.15])
  sess = tf.compat.v1.keras.backend.get_session()
  try:
    sess.run(tf.compat.v1.initializers.tables_initializer(name='init_all_tables'))
  except:
    print('Table init execption')
  if(hparams['resume'] and os.path.exists(hparams['checkPointDir']+'_restore.h5')):
    model.load_weights(hparams['checkPointDir']+'_restore.h5')
    print('Model restored')
# Start training/evaluation
#model.fit(x = trainData, epochs = hparams['epochs'], verbose = 2, callbacks = callbacksList, steps_per_epoch = hparams['steps_per_epoch'])
model.fit(x = trainData, epochs = hparams['epochs'], verbose = 2, callbacks = callbacksList, validation_data=evalData, validation_steps = hparams['validation_steps'], validation_freq=hparams['validation_freq'], steps_per_epoch = hparams['steps_per_epoch'])