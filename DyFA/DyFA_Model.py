from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, UpSampling3D
import tensorflow as tf
from config import*
from loss_funnction_And_matrics import*
import numpy as np
from DenseVnet3D import DenseVnet3D
#from Unet3D import Unet3D

####----Residual Blocks used for Resnet3D
def Residual_Block(inputs,
                 out_filters,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 use_bias=False,
                 activation=tf.nn.relu6,
                 kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform'),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                 bias_regularizer=None,
                 **kwargs):


    conv_params={'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    in_filters = inputs.get_shape().as_list()[-1]
    x=inputs
    orig_x=x

    ##building
    # Adjust the strided conv kernel size to prevent losing information
    k = [s * 2 if s > 1 else k for k, s in zip(kernel_size, strides)]

    if np.prod(strides) != 1:
            orig_x = tf.keras.layers.MaxPool3D(pool_size=strides,strides=strides,padding='valid')(orig_x)

    ##sub-unit-0
    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    x=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=k,strides=strides,**conv_params)(x)

    ##sub-unit-1
    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    x=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=kernel_size,strides=(1,1,1),**conv_params)(x)

        # Handle differences in input and output filter sizes
    if in_filters < out_filters:
        orig_x = tf.pad(tensor=orig_x,paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                    int(np.floor((out_filters - in_filters) / 2.)),
                    int(np.ceil((out_filters - in_filters) / 2.))]])

    elif in_filters > out_filters:
        orig_x = tf.keras.layers.Conv3D(filters=out_filters,kernel_size=kernel_size,strides=(1,1,1),**conv_params)(orig_x)

    x += orig_x
    return x



## Resnet----3D
def Resnet3D(inputs,
              num_classes,
              num_res_units=TRAIN_NUM_RES_UNIT,
              filters=TRAIN_NUM_FILTERS,
              strides=TRAIN_STRIDES,
              use_bias=False,
              activation=TRAIN_CLASSIFY_ACTICATION,
              kernel_initializer=TRAIN_KERNAL_INITIALIZER,
              bias_initializer=tf.zeros_initializer(),
              kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
              bias_regularizer=None,
              **kwargs):
    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}


    ##building
    k = [s * 2 if s > 1 else 3 for s in strides[0]]


    #Input
    x = inputs
    #1st-convo
    x=tf.keras.layers.Conv3D(filters[0], k, strides[0], **conv_params)(x)

    for res_scale in range(1, len(filters)):
        x = Residual_Block(
                inputs=x,
                out_filters=filters[res_scale],
                strides=strides[res_scale],
                activation=activation,
                name='unit_{}_0'.format(res_scale))
        for i in range(1, num_res_units):
            x = Residual_Block(
                    inputs=x,
                    out_filters=filters[res_scale],
                    strides=(1, 1, 1),
                    activation=activation,
                    name='unit_{}_{}'.format(res_scale, i))


    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    #axis = tuple(range(len(x.get_shape().as_list())))[1:-1]
    #x = tf.reduce_mean(x, axis=axis, name='global_avg_pool')
    x=tf.keras.layers.GlobalAveragePooling3D()(x)
    x =tf.keras.layers.Dropout(0.5)(x)
    classifier=tf.keras.layers.Dense(units=num_classes,activation='sigmoid')(x)

    #model = tf.keras.Model(inputs=inputs, outputs=classifier)
    #model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss=[TRAIN_CLASSIFY_LOSS], metrics=[TRAIN_CLASSIFY_METRICS,tf.keras.metrics.AUC()])

    return classifier

### Final Model
def DyFAModel_WithUnet(Unet_Model_Path,Input_shape,num_classes_clf,num_classes_for_seg):

    ###----Loading Segmentation Module---###
    inputs = tf.keras.Input(shape=Input_shape, name='CT')
    model_3DUnet=Unet3D(inputs,num_classes_for_seg)

    #-| Loading the Best Segmentation Weight
    model_3DUnet.load_weights(Unet_Model_Path)
    #-| Making the Segmentation Model Non-Trainable
    model_3DUnet.trainable = False

    #--| Getting the Features from Different Resolutions
    f_r1=(model_3DUnet.get_layer('Feature_R1').output)
    f_r2=(model_3DUnet.get_layer('Feature_R2').output)
    f_r3=(model_3DUnet.get_layer('Feature_R3').output)
    f_r4=(model_3DUnet.get_layer('Feature_R4').output)
    #f_r5=(model_3DUnet.get_layer('Feature_R5').output)
    last_predict=(model_3DUnet.get_layer('conv3d_17').output)
    #-| Upsampling the lower Resolution FA
    up2=(UpSampling3D(size = (2,2,2))(f_r2))
    up3=(UpSampling3D(size = (4,4,4))(f_r3))
    up4=(UpSampling3D(size = (8,8,8))(f_r4))
    #up5=(UpSampling3D(size = (16,16,16))(f_r5))
    #-| Concatenate the FAs
    FA_concatination=concatenate([f_r1,up2,up3,up4,last_predict],axis=-1)

    #-|| DyFA- Pass the Concatinated Feature to 1x1x1 convolution to get a 1 channel Volume.
    DyFA=tf.keras.layers.Conv3D(1, 1, name='DyFA')(FA_concatination)

    #-|| Making a HxWxDx2 channel Input data for the DyFA Classification Model
    DyFA_INPUT=concatenate([DyFA,inputs],axis=-1)

    DyFA_Model_output=Resnet3D(DyFA_INPUT,num_classes=num_classes_clf)

    Final_DyFAmodel=tf.keras.Model(inputs=inputs, outputs=DyFA_Model_output)

    return Final_DyFAmodel


def DyFAModel_withDenseVnet(DenseVnet3D_Model_Path,Input_shape,num_classes_clf,num_classes_for_seg):

    ###----Loading Segmentation Module---###
    inputs = tf.keras.Input(shape=Input_shape, name='CT')
    model_3DDenseVnet=DenseVnet3D(inputs,nb_classes=SEG_NUMBER_OF_CLASSES,encoder_nb_layers=NUM_DENSEBLOCK_EACH_RESOLUTION,growth_rate=NUM_OF_FILTER_EACH_RESOLUTION,dilation_list=DILATION_RATE,dropout_rate=DROPOUT_RATE)    
    #-| Loading the Best Segmentation Weight
    model_3DDenseVnet.load_weights(DenseVnet3D_Model_Path)
    model_3DDenseVnet.summary()
    #-| Making the Segmentation Model Non-Trainable
    model_3DDenseVnet.trainable = False
    #-| Getting the features
    f_60_192_96_96=(model_3DDenseVnet.get_layer('concatenate_25').output)
    #last_predict=(model_3DDenseVnet.get_layer('conv3d_63').output)
    #-| Upsampling the lower Resolution FA
    upsampled_F=(UpSampling3D(size = (2,2,2))(f_60_192_96_96))
    #-| Concatenate the FAs
    #FA_concatination=concatenate([upsampled_F,last_predict],axis=-1)

    #-|| DyFA- Pass the Concatinated Feature to 1x1x1 convolution to get a 1 channel Volume.
    DyFA=tf.keras.layers.Conv3D(1, 1, name='DyFA')(upsampled_F)

    #-|| Making a HxWxDx2 channel Input data for the DyFA Classification Model
    DyFA_INPUT=concatenate([DyFA,inputs],axis=-1)

    DyFA_Model_output=Resnet3D(DyFA_INPUT,num_classes=num_classes_clf)

    Final_DyFAmodel=tf.keras.Model(inputs=inputs, outputs=DyFA_Model_output)

    return Final_DyFAmodel
