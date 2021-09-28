

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, ReLU, Input, Add, UpSampling2D
from tensorflow.keras import activations

import numpy as np


def conv_block(x, filters, kernel=(3,3), stride=(1,1), do_relu=False): #FIXME does this need relu or anything?
    
    #Single basic convolution
    x = Conv2D(filters=filters, kernel_size=kernel, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)

    if do_relu:
        x = ReLU()(x)

    return x


def depthwise_separable_conv_block(x, filters, kernel=(3, 3), stride=(1,1)):
    
    # Depthwise convolution (one for each channel)
    x = DepthwiseConv2D(kernel_size=kernel, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    # No ReLU due to little observed effect

    # Pointwise convolution (1x1) for actual new features
    x = Conv2D(filters, kernel_size=(1,1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x


def bottleneck_res_block_part(x, filters, expansion_factor, kernel, stride):
    # Keeping shortcut to start for eventual joining
    x_shortcut = x
    
    ## Getting number of channels in input x (source: https://github.com/xiaochus/MobileNetV2)
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    # Depth
    initial_channels = keras.backend.int_shape(x)[channel_axis]
    expanded_channels = initial_channels * expansion_factor

    x = Conv2D(filters=expanded_channels, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    ## DEPTHWISE LAYER - 3x3 depthwise convolution layer
    x = DepthwiseConv2D(kernel_size=kernel, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # PROJECTION LAYER - 1x1 pointwise convolution layer
    x = Conv2D(filters=filters, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)

    # RESIDUAL CONNECTION
    # Create the residual connection if the number of input/output channels (and input/output shape) is the same
    if initial_channels == filters and stride == (1,1): #FIXME GET THIS CHECKED
        x = keras.layers.Add()([x_shortcut, x])

    return x


def bottleneck_res_block(x, filters, expansion_factor, kernel=(3, 3), stride=(1, 1), repeats=1): 

    x = bottleneck_res_block_part(x, filters, expansion_factor, kernel, stride)

    # Performing repetitions (with stride (1,1), guaranteeing residual connection)
    for i in range(1, repeats):
        x = bottleneck_res_block(x, filters, expansion_factor, kernel, (1,1))

    return x

def pyramid_pooling_block(input_tensor, bin_sizes, input_size):
  concat_list = [input_tensor]
  w = input_size[0] // 32
  h = input_size[1] // 32

  for bin_size in bin_sizes:
    x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
    x = tf.keras.layers.Conv2D(32, 3, 2, padding='same',kernel_regularizer=keras.regularizers.l2(0.00004), bias_regularizer=keras.regularizers.l2(0.00004))(x)
    x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(x)

    concat_list.append(x)

  return tf.keras.layers.concatenate(concat_list)

def model(num_classes=19, input_size=(1024, 2048, 3), shrink_factor=4):

    ## DEEP BRANCH
    # Reference: Table on page 5 of paper
    reduced_input_size = (int(input_size[0]/shrink_factor), int(input_size[1]/shrink_factor), input_size[2])

    deep_input = Input(shape=reduced_input_size, name='input_deep')
    deep_branch = conv_block(deep_input, filters=32, kernel=(3,3), stride=(2,2), do_relu=True)
    deep_branch = bottleneck_res_block(deep_branch, filters=24, expansion_factor=1, repeats=1) 
    deep_branch = bottleneck_res_block(deep_branch, filters=32, expansion_factor=6, repeats=1)
    deep_branch = bottleneck_res_block(deep_branch, filters=48, expansion_factor=6, repeats=3, stride=(2,2))
    deep_branch = bottleneck_res_block(deep_branch, filters=64, expansion_factor=6, repeats=3, stride=(2,2)) 
    deep_branch = bottleneck_res_block(deep_branch, filters=96, expansion_factor=6, repeats=2)
    deep_branch = bottleneck_res_block(deep_branch, filters=128, expansion_factor=6, repeats=2)
    
    #PPM
    deep_branch = pyramid_pooling_block(deep_branch, [2,4,6,8], input_size)
    
    deep_branch = conv_block(deep_branch, filters=128, kernel=(1,1), do_relu=True) 
    ## SHALLOW BRANCH

    shallow_input = Input(shape=input_size, name='input_shallow')
    shallow_branch = conv_block(shallow_input, filters=32, kernel=(3,3), stride=(2,2))
    downsample = shallow_branch
    shallow_branch = depthwise_separable_conv_block(shallow_branch, filters=48, kernel=(3,3), stride=(2, 2))
    shallow_branch = depthwise_separable_conv_block(shallow_branch, filters=64, kernel=(3,3), stride=(2, 2)) 
    shallow_branch = depthwise_separable_conv_block(shallow_branch, filters=96, kernel=(3,3), stride=(1, 1))
    
    ## FEATURE FUSION 
    # Deep branch prep
    deep_branch = UpSampling2D((4, 4))(deep_branch) 
    deep_branch = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(4,4), padding='same')(deep_branch)
    deep_branch = tf.keras.layers.BatchNormalization()(deep_branch) 
    deep_branch = tf.keras.activations.relu(deep_branch) 
    deep_branch = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same')(deep_branch) 

    # Shallow branch prep
    shallow_branch = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same')(shallow_branch)
    
    # Actual addition
    output = Add()([shallow_branch, deep_branch])
    output = tf.keras.activations.relu(output) 

    
    # Dropout layer before final softmax
    output = tf.keras.layers.Dropout(rate=0.35)(output)

    # Final result using number of classes
    output = Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), activation='softmax', name='conv_output')(output)

    # Perform upsample to return to original resolution NOTE: Not in original paper
    output = UpSampling2D((8, 8))(output)

    ## MAKING MODEL
    contextnet = Model(inputs=[shallow_input, deep_input], outputs=output)

    return contextnet