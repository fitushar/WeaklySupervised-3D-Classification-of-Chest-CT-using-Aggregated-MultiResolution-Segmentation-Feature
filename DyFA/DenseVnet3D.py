from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf



##########---tf bilinear UpSampling3D
def up_sampling(input_tensor, scale):
    net = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(scale, scale), interpolation='bilinear'))(input_tensor)
    net = tf.keras.layers.Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
    net = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(scale, 1), interpolation='bilinear'))(net)
    net = tf.keras.layers.Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
    return net

#######-----Bottleneck
def Bottleneck(x, nb_filter, increase_factor=4., weight_decay=1e-4):
    inter_channel = int(nb_filter * increase_factor)
    x = tf.keras.layers.Conv3D(inter_channel, (1, 1, 1),
                               kernel_initializer='he_normal',
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(x)
    x = tf.nn.relu6(x)
    return x

#####------------>>> Convolutional Block
def conv_block(input, nb_filter, kernal_size=(3, 3, 3), dilation_rate=1,
                 bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3X3 Conv3D, optional bottleneck block and dropout
    Args:
        input: Input tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: tensor with batch_norm, relu and convolution3D added (optional bottleneck)
    '''


    x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(input)
    x = tf.nn.relu6(x)

    if bottleneck:
        inter_channel = nb_filter  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
        x = tf.keras.layers.Conv3D(inter_channel, (1, 1, 1),
                   kernel_initializer='he_normal',
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(x)
        x = tf.nn.relu6(x)

    x = tf.keras.layers.Conv3D(nb_filter, kernal_size,
               dilation_rate=dilation_rate,
               kernel_initializer='he_normal',
               padding='same',
               use_bias=False)(x)
    if dropout_rate:
        x = tf.keras.layers.SpatialDropout3D(dropout_rate)(x)
    return x

##--------------------DenseBlock-------####
def dense_block(x, nb_layers, growth_rate, kernal_size=(3, 3, 3),
                  dilation_list=None,
                  bottleneck=True, dropout_rate=None, weight_decay=1e-4,
                  return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: input tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: tensor with nb_layers of conv_block appended
    '''

    if dilation_list is None:
        dilation_list = [1] * nb_layers
    elif type(dilation_list) is int:
        dilation_list = [dilation_list] * nb_layers
    else:
        if len(dilation_list) != nb_layers:
            raise ('the length of dilation_list should be equal to nb_layers %d' % nb_layers)

    x_list = [x]

    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, kernal_size, dilation_list[i],
                          bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)
        if i == 0:
            x = cb
        else:
            x = tf.keras.layers.concatenate([x, cb], axis=-1)

    if return_concat_list:
        return x, x_list
    else:
        return x

###---------transition_block
def transition_block(input, nb_filter, compression=1.0, weight_decay=1e-4,
                       pool_kernal=(3, 3, 3), pool_strides=(2, 2, 2)):
    ''' Apply BatchNorm, Relu 1x1, Conv3D, optional compression, dropout and Maxpooling3D
    Args:
        input: input tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''


    x =tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(input)
    x = tf.nn.relu6(x)
    x = tf.keras.layers.Conv3D(int(nb_filter * compression), (1, 1, 1),
               kernel_initializer='he_normal',
               padding='same',
               use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.AveragePooling3D(pool_kernal, strides=pool_strides)(x)

    return x

###---Trasnsition up block
def transition_up_block(input, nb_filters, compression=1.0,
                          kernal_size=(3, 3, 3), pool_strides=(2, 2, 2),
                          type='deconv', weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        input: tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''

    if type == 'upsampling':
        x = tf.keras.layers.UpSampling3D(size=kernal_size, interpolation='bilinear')(input)
        x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(x)
        x = tf.nn.relu6(x)
        x = tf.keras.layers.Conv3D(int(nb_filters * compression), (1, 1, 1),
                   kernel_initializer='he_normal',
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

    else:
        x = tf.keras.layers.Conv3DTranspose(int(nb_filters * compression),
                            kernal_size,
                            strides=pool_strides,
                            activation='relu',
                            padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)

    return x



def DenseVnet3D(inputs,
                nb_classes=1,
                encoder_nb_layers=(5, 8, 8),
                growth_rate=(4, 8, 12),
                dilation_list=(5, 3, 1),
                dropout_rate=0.25,
                weight_decay=1e-4,
                init_conv_filters=24):
    """ 3D DenseVNet Implementation by f.i.tushar, tf 2.0.
        This is a tensorflow 2.0 Implementation of paper:
        Gibson et al., "Automatic multi-organ segmentation on abdominal CT with
        dense V-networks" 2018.

        Reference Implementation: vision4med :i) https://github.com/baibaidj/vision4med/blob/5c23f57c2836bfabd7bd95a024a0a0b776b181b5/nets/DenseVnet.py
                                             ii) https://niftynet.readthedocs.io/en/dev/_modules/niftynet/network/dense_vnet.html#DenseVNet

    Input
      |
      --[ DFS ]-----------------------[ Conv ]------------[ Conv ]------[+]-->
           |                                       |  |              |
           -----[ DFS ]---------------[ Conv ]------  |              |
                   |                                  |              |
                   -----[ DFS ]-------[ Conv ]---------              |
                                                          [ Prior ]---
    Args:
        inputs: Input , input shape should be (Batch,D,H,W,channels)
        nb_classes: number of classes
        encoder_nb_layers: Number of Layer in each dense_block
        growth_rate: Number of filters in each DenseBlock
        dilation_list=Dilation rate each level
        dropout_rate: dropout rate
        weight_decay: weight decay
    Returns: Returns the Segmentation Prediction of Given Input Shape
    """
    #--|Getting the Input
    img_input = inputs
    input_shape = tf.shape(img_input) # Input shape
    nb_dense_block = len(encoder_nb_layers)# Convert tuple to list

    # Initial convolution
    x = tf.keras.layers.Conv3D(init_conv_filters, (5, 5, 5),
               strides=2,
               kernel_initializer='he_normal',
               padding='same',
               name='initial_conv3D',
               use_bias=False,
               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(img_input)
    x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(x)
    x = tf.nn.relu6(x)

    #Making the skiplist for concationatin
    skip_list = []

    # Add dense blocks
    for block_idx in range(nb_dense_block):
        '''
        |--Input for dense_block is as following
        |---#x=Input,
            #encoder_nb_layers[block_idx]=Number of layer in a dense_block
            #growth_rate[block_idx]= Number of Filter in that DenseBlock
            #dilation_list= Dilation Rate.

        '''
        x = dense_block(x, encoder_nb_layers[block_idx],
                              growth_rate[block_idx],
                              kernal_size=(3, 3, 3),
                              dilation_list=dilation_list[block_idx],
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay,
                              )

        # Skip connection
        skip_list.append(x)
        #Pooling
        x = tf.keras.layers.AveragePooling3D((2, 2, 2))(x)
        # x = __transition_block(x, nb_filter,compression=compression,weight_decay=weight_decay,pool_kernal=(3, 3, 3),pool_strides=(2, 2, 2))


    ##Convolutiion and third Resolution layer and Updample.
    x_level3 = conv_block(skip_list[-1], growth_rate[2], bottleneck=True, dropout_rate=dropout_rate)
    x_level3 = up_sampling(x_level3, scale=4)
    # x_level3 = UpSampling3D(size = (4,4,4))(x_level3)

    ##Convolutiion and 2nd Resolution layer and Updample.
    x_level2 = conv_block(skip_list[-2], growth_rate[1], bottleneck=True, dropout_rate=dropout_rate)
    x_level2 = up_sampling(x_level2, scale=2)
    # x_level2 = UpSampling3D(size=(2, 2, 2))(x_level2)

    ##Convolutiion and first Resolution layer
    x_level1 = conv_block(skip_list[-3], growth_rate[0], bottleneck=True, dropout_rate=dropout_rate)
    #x_level1 = up_sampling(x_level1, scale=2)
    x = tf.keras.layers.Concatenate()([x_level3, x_level2, x_level1])

    ###--Final Convolution---
    x = conv_block(x, 24, bottleneck=False, dropout_rate=dropout_rate)
    ##----Upsampling--TheFinal Output----#####
    x = up_sampling(x, scale=2)

    ####------Prediction---------------###
    if nb_classes == 1:
        x = tf.keras.layers.Conv3D(nb_classes, 1, activation='sigmoid', padding='same', use_bias=False)(x)
    elif nb_classes > 1:
        x = tf.keras.layers.Conv3D(nb_classes, 1, activation='softmax', padding='same', use_bias=False)(x)
        #x = tf.argmax(x, axis=-1)
    print(x)

    # Create model.
    model = tf.keras.Model(img_input, x, name='DenseVnet3D')
    return model
'''
###################----Demo Usages----#############
INPUT_PATCH_SIZE=[384,192,192,1]
NUMBER_OF_CLASSES=1
inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')

#Model_3D=DenseVnet3D(inputs,nb_classes=1,encoder_nb_layers=(5, 8, 8),growth_rate=(4, 8, 12),dilation_list=(5, 3, 1))
Model_3D=DenseVnet3D(inputs,nb_classes=1,encoder_nb_layers=(4, 8, 16),growth_rate=(12,24,24),dilation_list=(5, 10, 10),dropout_rate=0.25)
Model_3D.summary()
tf.keras.utils.plot_model(Model_3D, 'DenseVnet3D.png',show_shapes=True)
'''
