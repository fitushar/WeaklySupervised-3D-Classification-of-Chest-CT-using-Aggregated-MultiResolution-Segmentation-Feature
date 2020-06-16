from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
'''
tf.config.optimizer.set_jit(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)
'''

from tensorflow.keras.optimizers import Adam
from config import*
import os
import datetime
from DyFA_Model import*
from tfrecords_utilities import decode_ct
import numpy as np
import random

####----Getting --the tfrecords
def getting_list(path):
    a=[file for file in os.listdir(path) if file.endswith('.tfrecords')]
    all_tfrecoeds=random.sample(a, len(a))
    #all_tfrecoeds.sort(key=lambda f: int(filter(str.isdigit, f)))
    list_of_tfrecords=[]
    for i in range(len(all_tfrecoeds)):
        tf_path=path+all_tfrecoeds[i]
        list_of_tfrecords.append(tf_path)
    return list_of_tfrecords

#--Traing Decoder
def load_training_tfrecords(record_mask_file,batch_size):
    dataset=tf.data.Dataset.list_files(record_mask_file).interleave(lambda x: tf.data.TFRecordDataset(x),cycle_length=NUMBER_OF_PARALLEL_CALL,num_parallel_calls=NUMBER_OF_PARALLEL_CALL)
    dataset=dataset.map(decode_ct,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).repeat(TRAING_EPOCH).batch(batch_size)
    batched_dataset=dataset.prefetch(PARSHING)
    return batched_dataset

#--Validation Decoder
def load_validation_tfrecords(record_mask_file,batch_size):
    dataset=tf.data.Dataset.list_files(record_mask_file).interleave(tf.data.TFRecordDataset,cycle_length=NUMBER_OF_PARALLEL_CALL,num_parallel_calls=NUMBER_OF_PARALLEL_CALL)
    dataset=dataset.map(decode_ct,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).repeat(TRAING_EPOCH).batch(batch_size)
    batched_dataset=dataset.prefetch(PARSHING)
    return batched_dataset


def Training():

    #TensorBoard
    logdir = os.path.join(LOG_NAME, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    ##csv_logger
    csv_logger = tf.keras.callbacks.CSVLogger(TRAINING_CSV)
    ##Model-checkpoings
    path=TRAINING_SAVE_MODEL_PATH
    model_path=os.path.join(path, MODEL_SAVING_NAME)
    Model_callback= tf.keras.callbacks.ModelCheckpoint(filepath=model_path,save_best_only=False,save_weights_only=True,monitor=ModelCheckpoint_MOTITOR,verbose=1)
    ##----Preparing Data
    tf_train=getting_list(TRAINING_TF_RECORDS)
    tf_val=getting_list(VALIDATION_TF_RECORDS)
    traing_data=load_training_tfrecords(tf_train,BATCH_SIZE)
    Val_batched_dataset=load_validation_tfrecords(tf_val,BATCH_SIZE)

    if (NUM_OF_GPU==1):
        if RESUME_TRAINING==1:
            inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
            Model_3D=DyFAModel_withDenseVnet(SEGMENTATION_MODEL_PATH,INPUT_PATCH_SIZE,NUMBER_OF_CLASSES,SEGMENTATION_NUM_OF_CLASSES)
            Model_3D.load_weights(RESUME_TRAIING_MODEL)
            initial_epoch_of_training=TRAINING_INITIAL_EPOCH
            print('Resume-Training From-Epoch{}-Loading-Model-from_{}'.format(initial_epoch_of_training,RESUME_TRAIING_MODEL))
            Model_3D.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
            Model_3D.summary()
        else:
            initial_epoch_of_training=0
            inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
            Model_3D=DyFAModel_withDenseVnet(SEGMENTATION_MODEL_PATH,INPUT_PATCH_SIZE,NUMBER_OF_CLASSES,SEGMENTATION_NUM_OF_CLASSES)
            Model_3D.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[TRAIN_CLASSIFY_METRICS,tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
            Model_3D.summary()
        Model_3D.fit(traing_data,
                   steps_per_epoch=TRAINING_STEP_PER_EPOCH,
                   epochs=TRAING_EPOCH,
                   initial_epoch=initial_epoch_of_training,
                   validation_data=Val_batched_dataset,
                   validation_steps=VALIDATION_STEP,
                   callbacks=[tensorboard_callback,csv_logger,Model_callback])

    ###Multigpu----
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy(DISTRIIBUTED_STRATEGY_GPUS)
        with mirrored_strategy.scope():
                if RESUME_TRAINING==1:
                    inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
                    Model_3D=DyFAModel_withDenseVnet(SEGMENTATION_MODEL_PATH,INPUT_PATCH_SIZE,NUMBER_OF_CLASSES,SEGMENTATION_NUM_OF_CLASSES)
                    Model_3D.load_weights(RESUME_TRAIING_MODEL)
                    initial_epoch_of_training=TRAINING_INITIAL_EPOCH
                    print('Resume-Training From-Epoch{}-Loading-Model-from_{}'.format(initial_epoch_of_training,RESUME_TRAIING_MODEL))
                    Model_3D.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
                    Model_3D.summary()
                else:
                    initial_epoch_of_training=0
                    inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
                    Model_3D=DyFAModel_withDenseVnet(SEGMENTATION_MODEL_PATH,INPUT_PATCH_SIZE,NUMBER_OF_CLASSES,SEGMENTATION_NUM_OF_CLASSES)
                    Model_3D.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[TRAIN_CLASSIFY_METRICS,tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
                    Model_3D.summary()
                Model_3D.fit(traing_data,
                           steps_per_epoch=TRAINING_STEP_PER_EPOCH,
                           epochs=TRAING_EPOCH,
                           initial_epoch=initial_epoch_of_training,
                           validation_data=Val_batched_dataset,
                           validation_steps=VALIDATION_STEP,
                           callbacks=[tensorboard_callback,csv_logger,Model_callback])

if __name__ == '__main__':
   Training()
