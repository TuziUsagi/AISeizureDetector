#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from inception import eeg_inception_gpu as create_model
from readDataset import readDataset
from hparam import gethparam
from tensorflow.python import debug as tf_debug

#Setup network
tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
hparams = gethparam()

# Get dataset
trainData = readDataset(hparams, 'Train', hparams['batch_size'])
evalData = readDataset(hparams, 'Eval', hparams['batch_size']>>4)
hparams['validation_steps'] = hparams['validation_steps'] << 4

# Get optimizer
EEGOptimizer = tf.keras.optimizers.Adam(learning_rate = hparams['learning_rate'], beta_1 = hparams['AdamBeta1'], beta_2 = hparams['AdamBeta2'], epsilon = hparams['epsilon'])
# Enable mixed precision to take advantage of tensorCore
if(hparams['mixedPrecision']):
  #tf.compat.v1.keras.backend.set_floatx('float16')
  EEGOptimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(EEGOptimizer)
 
# Get loss function
lossFunc = tf.keras.losses.CategoricalCrossentropy()

# Metrics
accMetrics = tf.keras.metrics.CategoricalAccuracy()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=hparams['logDir'],write_grads = True, histogram_freq = 10, batch_size = hparams['batch_size'], update_freq = 'epoch', profile_batch=0, write_images=True)
checkpointWeight_callback = tf.keras.callbacks.ModelCheckpoint(hparams['checkPointDir']+'.h5', save_best_only=True, load_weights_on_restart=False, monitor = hparams['monitorVal'], save_weights_only = True)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(hparams['checkPointDir']+'_restore.h5', save_best_only=False, load_weights_on_restart=hparams['resume'], monitor = hparams['monitorVal'], save_weights_only = False)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, verbose = 1, min_delta = 0.001, cooldown = 3, min_lr=0.0005)
callbacksList = [tensorboard_callback, checkpointWeight_callback, reduce_lr, checkpoint_callback]

# Get model
model = create_model(hparams, 'Train')
model.summary()
# compile model
model.compile(optimizer = EEGOptimizer, loss = lossFunc, metrics = [accMetrics], loss_weights=[1., 0.15, 0.15, 0.15, 0.15])

# Start training/evaluation
sess = tf.compat.v1.keras.backend.get_session()
sess.run(tf.initializers.global_variables())
try:
  sess.run(tf.compat.v1.initializers.tables_initializer(name='init_all_tables'))
except:
  print('Table init execption')
if(hparams['resume'] and os.path.exists(hparams['checkPointDir']+'_restore.h5')):
  model.load_weights(hparams['checkPointDir']+'_restore.h5')
  print('Model restored')
#keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6007"))
model.fit(x = trainData, epochs = hparams['epochs'], verbose = 2, callbacks = callbacksList, validation_data=evalData, validation_steps = hparams['validation_steps'], validation_freq=hparams['validation_freq'], steps_per_epoch = hparams['steps_per_epoch'])
