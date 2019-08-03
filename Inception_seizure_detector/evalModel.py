#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inception_tiny import eeg_inception_tpu as create_model
from readDataset import readDataset
from hparam import gethparam
from tensorflow.python import debug as tf_debug
import os
from tensorflow.compat.v1.keras import backend as K

# Disable learning, all variables will be treated as constant
K.set_learning_phase(0)
#Setup network
hparams = gethparam()
hparams['64bFilter'] = False
hparams['mixedPrecision'] = False
# Get dataset
#trainData = readDataset(hparams, 'Train')
evalData = readDataset(hparams, 'Eval')

# Get optimizer
EEGOptimizer = tf.keras.optimizers.Adam(learning_rate = hparams['learning_rate'], beta_1 = hparams['AdamBeta1'], beta_2 = hparams['AdamBeta2'], epsilon = hparams['epsilon'])
 
# Get loss function
lossFunc = tf.keras.losses.CategoricalCrossentropy()

# Metrics
accMetrics = tf.keras.metrics.CategoricalAccuracy()

checkpointWeight_callback = tf.keras.callbacks.ModelCheckpoint(hparams['checkPointDir']+'.h5', save_best_only=True, load_weights_on_restart=False, monitor = 'val_loss', save_weights_only = True)
callbacksList = [checkpointWeight_callback]

# Get model
model = create_model(hparams, 'Eval', hparams['batch_size])
model.summary()
model.compile(optimizer = EEGOptimizer, loss = lossFunc, metrics = [accMetrics],  loss_weights=[1., 0.2, 0.2, 0.2, 0.2])
sess = tf.compat.v1.keras.backend.get_session()
try:
  sess.run(tf.compat.v1.initializers.tables_initializer(name='init_all_tables'))
except:
  print('Table init execption')
model.load_weights(hparams['checkPointDir']+'.h5')
# Start training/evaluation
a = model.evaluate(x = evalData, verbose = 1, callbacks = callbacksList, steps = hparams['validation_steps'])
