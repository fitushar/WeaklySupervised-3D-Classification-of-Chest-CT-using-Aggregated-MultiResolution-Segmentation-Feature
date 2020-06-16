import tensorflow as tf
from loss_funnction_And_matrics import*
import math
###---Number-of-GPU
NUM_OF_GPU=1
#["gpu:1","gpu:2","gpu:3"]
DISTRIIBUTED_STRATEGY_GPUS=["gpu:0"]
###-----SEGMENATTION----###
SEGMENTATION_MODEL_PATH='/image_data/Scripts/April_Model/DyFA_61FAvg_April17_2020/LungSEG_DenseVnet_2.60_4998.h5'
SEGMENTATION_NUM_OF_CLASSES=31
#####-----Configure DenseVnet3D---##########
SEG_NUMBER_OF_CLASSES=31
SEG_INPUT_PATCH_SIZE=(128,160,160, 1)
NUM_DENSEBLOCK_EACH_RESOLUTION=(4, 8, 16)
NUM_OF_FILTER_EACH_RESOLUTION=(12,24,24)
DILATION_RATE=(5, 10, 10)
DROPOUT_RATE=0.25
###----Resume-Training
RESUME_TRAINING=1
RESUME_TRAIING_MODEL='/image_data/Scripts/April_Model/DyFA_61FAvg_April17_2020/Model_DyFA_61FAvg_April17_2020/DyFAModel60FAvg_9.62_55.h5'
TRAINING_INITIAL_EPOCH=55
##Network Configuration
NUMBER_OF_CLASSES=5
INPUT_PATCH_SIZE=(128,160,160, 1)
TRAIN_NUM_RES_UNIT=3
TRAIN_NUM_FILTERS=(16, 32, 64, 128)
TRAIN_STRIDES=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2))
TRAIN_CLASSIFY_ACTICATION=tf.nn.relu6
TRAIN_KERNAL_INITIALIZER=tf.keras.initializers.VarianceScaling(distribution='uniform')
##Training Hyper-Parameter
##Training Hyper-Parameter
TRAIN_CLASSIFY_LEARNING_RATE =1e-4
TRAIN_CLASSIFY_LOSS=Weighted_BCTL
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)
TRAIN_CLASSIFY_METRICS=tf.keras.metrics.AUC()
BATCH_SIZE=6
TRAINING_STEP_PER_EPOCH=math.ceil((3514)/BATCH_SIZE)
VALIDATION_STEP=math.ceil((759)/BATCH_SIZE)
TRAING_EPOCH=300
NUMBER_OF_PARALLEL_CALL=3
PARSHING=2*BATCH_SIZE
#--Callbacks-----
ModelCheckpoint_MOTITOR='val_loss'
TRAINING_SAVE_MODEL_PATH='/image_data/Scripts/April_Model/DyFA_61FAvg_April17_2020/Model_DyFA_61FAvg_April17_2020/'
TRAINING_CSV='DyFA_61FAvg_April17_2020.csv'
LOG_NAME="Log_DyFA_60FAvg_April17_2020"
MODEL_SAVING_NAME="DyFAModel60FAvg_{val_loss:.2f}_{epoch}.h5"

####
TRAINING_TF_RECORDS='/image_data/nobackup/Lung_CenterPatch_2mm_March27_2020/tf/Train_tfrecords/'
VALIDATION_TF_RECORDS='/image_data/nobackup/Lung_CenterPatch_2mm_March27_2020/tf/Val_tfrecords/'
