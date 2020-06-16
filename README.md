# Weakly-Supervised-3D-Classification-of-Chest-CT-using-Aggregated-Multi-Resolution-Deep-Segmentation-
This Repo contains the updated implementation of our paper "Weakly supervised 3D classification of chest CT using aggregated multi-resolution deep segmentation features", Proc. SPIE 11314, Medical Imaging 2020: Computer-Aided Diagnosis, 1131408 (16 March 2020).

* Version-1: Implemented Segmentation Module and Classification Seperately and was in Tensorflow 1.x
Can be seen here: https://github.com/anindox8/Deep-Segmentation-Features-for-Weakly-Supervised-3D-Disease-Classification-in-Chest-CT

* Version-2: Updated the Implementation , For reducing computation expenses the Segmenation Module and Classifiction Module is combined,updated implementation is in Tensorflow 2.0. This implemnetation is 2 times faster than the Version-1 in terms of training. Also Project has been moved from multi-class to multi-label classification setup (Follow the SPIE presentation for clear idea).

If our work help in your task or project please site the work at  (https://doi.org/10.1117/12.2550857). This work is been presented at  SPIE Medical Imaging, 2020, Houston, Texas, United States. Presentation can be seen here : https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11314/2550857/Weakly-supervised-3D-classification-of-chest-CT-using-aggregated-multi/10.1117/12.2550857.full?SSO=1

![model Architecture for multi-label approach version-2 implementation](https://github.com/fitushar/Weakly-Supervised-3D-Classification-of-Chest-CT-using-Aggregated-Multi-Resolution-Deep-Segmentation-/blob/master/figure/Model_Architecture.png)

## Citation
```
Anindo Saha*, Fakrul I. Tushar*, Khrystyna Faryna, Vincent M. D'Anniballe, Rui Hou, 
Maciej A. Mazurowski, Geoffrey D. Rubin M.D., and Joseph Y. Lo 
"Weakly supervised 3D classification of chest CT using aggregated multi-resolution deep segmentation features", 
Proc. SPIE 11314, Medical Imaging 2020: Computer-Aided Diagnosis,1131408 (16 March 2020); 
https://doi.org/10.1117/12.2550857
(*Authors with equal contribution to this work.)
```

## Directories and Files
*   i) DyFA -|--> Dynamic Feature aggragation Model and training script.
       ```ruby
             a) config.py |-- Configuration file to train the DyFA model
             b) DenseVnet3D.py |-- 3D implementation of the DenseVnet (Segmentation Module)
             c) DyFA_Model.py  |-- DyFA model (Segmentation+Classification Module)
             d) loss_funnction_And_matrics |-- Loss Function.
             e) resume_training_using_check_point |-- Training Script
             f) tfrecords_utilities |-- Tfrecords decoding function    
      ```       
*  ii) SyFa -|--> Static Feature aggragation Model and training script.
      ```ruby
             a) config.py |-- Configuration file to train the DyFA model
             b) DenseVnet3D.py |-- 3D implementation of the DenseVnet (Segmentation Module)
             c) DyFA_Model.py  |-- StFA model (Segmentation+Classification Module)
             d) loss_funnction_And_matrics |-- Loss Function.
             e) resume_training_using_check_point |-- Training Script
             f) tfrecords_utilities |-- Tfrecords decoding function    
      ```
* iii) Figure -|--> Figure used in this Repo
*  iv) SPIE_presentation -|--> SPIE presentation


## How to run

To run the model all is to need to configure the `config.py` based on your requiremnet. and use the command 
`python resume_training_using_check_point.py`

```ruby
import tensorflow as tf
from loss_funnction_And_matrics import*
import math
###---Number-of-GPU
NUM_OF_GPU=2
#["gpu:1","gpu:2","gpu:3"]
DISTRIIBUTED_STRATEGY_GPUS=["gpu:0","gpu:1"]
###-----SEGMENATTION----###
SEGMENTATION_MODEL_PATH='/Path/of/the/Segmentation Module/weight/Model.h5'.h5'
SEGMENTATION_NUM_OF_CLASSES=31
#####-----Configure DenseVnet3D---##########
SEG_NUMBER_OF_CLASSES=31
SEG_INPUT_PATCH_SIZE=(128,160,160, 1)
NUM_DENSEBLOCK_EACH_RESOLUTION=(4, 8, 16)
NUM_OF_FILTER_EACH_RESOLUTION=(12,24,24)
DILATION_RATE=(5, 10, 10)
DROPOUT_RATE=0.25
###----Resume-Training
'''
if want to resume training from the weights Set
RESUME_TRAINING=1
'''
RESUME_TRAINING=0
RESUME_TRAIING_MODEL='/Path/of/the/model/weight/Model.h5'
TRAINING_INITIAL_EPOCH=0
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
BATCH_SIZE=12
TRAINING_STEP_PER_EPOCH=math.ceil((3514)/BATCH_SIZE)
VALIDATION_STEP=math.ceil((759)/BATCH_SIZE)
TRAING_EPOCH=300
NUMBER_OF_PARALLEL_CALL=6
PARSHING=3*BATCH_SIZE
#--Callbacks-----
ModelCheckpoint_MOTITOR='val_loss'
TRAINING_SAVE_MODEL_PATH='/Path/to/save/model/weight/Model.h5'
TRAINING_CSV='DyFA_61FC1X1_April17_2020.csv'
LOG_NAME="Log_DyFA_61FC1X1_April17_2020"
MODEL_SAVING_NAME="DyFAModel61FC1X1_{val_loss:.2f}_{epoch}.h5"
####
TRAINING_TF_RECORDS='/Training/tfrecords/path/'
VALIDATION_TF_RECORDS='/Val/tfrecords/path/'
```

## Multi-label Data Statistics 
![Multi-label Data Statistics](https://github.com/fitushar/Weakly-Supervised-3D-Classification-of-Chest-CT-using-Aggregated-Multi-Resolution-Deep-Segmentation-/blob/master/figure/dataset.png)

## Results
![Classification Results](https://github.com/fitushar/Weakly-Supervised-3D-Classification-of-Chest-CT-using-Aggregated-Multi-Resolution-Deep-Segmentation-/blob/master/figure/results.png)


