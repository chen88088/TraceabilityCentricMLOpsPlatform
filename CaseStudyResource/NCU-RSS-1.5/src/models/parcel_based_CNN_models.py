# -*- coding: utf-8 -*-


import sys

import tensorflow as tf

from tensorflow.keras import layers
import logging
import numpy as np




# from keras.models 改成 from tensorflow.keras.models !!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from configs.logger_config import LOGGING_CONFIG
from logging import config

# VGG16
config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("train_info_log")

root_logger = logging.getLogger("train_debug_log")


def get_conv_weights_from_pretrained_VGG16():
    VGG16_conv_blocks = tf.keras.applications.VGG16(
        weights='imagenet', include_top=False)
    conv_layer_weight_list = []
    for layer in VGG16_conv_blocks.layers:
        if "conv" in layer.name:
            layer_w = layer.get_weights()
            conv_layer_weight_list.append(layer_w)

    return conv_layer_weight_list


def set_conv_weights_to_VGG16_conv_layers(
        model, conv_layer_weight_list, channel=3):
    pretrained_conv_weight_index = -1
    for layer in model.layers:
        if "conv2d" in layer.name:  # pretrained VGG16 backbone只有conv2d layer有網路權重
            pretrained_conv_weight_index += 1
            if pretrained_conv_weight_index == 0:  # 第一層conv2d layer

                # 若輸入影像為四通道影像，需要在給pretrained的第一層conv2d layer增加一個通道空間的權重
                if channel == 4:
                    pretrained_layer_weights = conv_layer_weight_list[pretrained_conv_weight_index]
                    kernel_weight = pretrained_layer_weights[0]
                    bias_weight = pretrained_layer_weights[1]

                    n_channel = kernel_weight.shape[2]

                    four_channel_kernel_weight = np.zeros(
                        (kernel_weight.shape[0], kernel_weight.shape[1], n_channel + 1, kernel_weight.shape[3]))
                    four_channel_bias_weight = np.zeros((n_channel + 1))

                    # Input layer後的第一個pretrained layer只有3通道weight,
                    # 取出Keras預設初始化的權重
                    initialized_new_layer_weight = layer.weights[0][:, :, 3, :]
                    # 放入pretrained layer的3通道weight
                    four_channel_kernel_weight[:, :, 0:n_channel, :] = kernel_weight
                    # 放入Keras預設初始化的權重
                    four_channel_kernel_weight[:, :, 3, :] = initialized_new_layer_weight
                    four_channel_bias_weight = bias_weight

                    four_channel_weights = [
                        four_channel_kernel_weight, four_channel_bias_weight]
                    layer.set_weights(four_channel_weights)

                else:  # 若輸入影像為三通道影像，不需要對pretrained 3 channel VGG16做任何改動
                    layer.set_weights(
                        conv_layer_weight_list[pretrained_conv_weight_index])
            else:
                # layer.set_weights(conv_layer_weight_list[pretrained_conv_weight_index])
                # logger.info("layer.weights[0].shape:", layer.weights[0].shape)
                # logger.info("conv_layer_weight_list[pretrained_conv_weight_index][0].shape:",conv_layer_weight_list[pretrained_conv_weight_index][0].shape)

                layer.set_weights(
                    conv_layer_weight_list[pretrained_conv_weight_index])

    if pretrained_conv_weight_index != 12:
        logger.info(" *** pretrained_conv_weight_index != 12 ")
        sys.exit(1)
    return model


### selected model
def pretrained_VGG16BNConv_GAP(
        input_shape=(500, 450, 3),
        ConvBlocks_BN=False,
        DP_rate=0,
        conv_blocks_trainable=False,
        L2reg=False,
        ConvBlocks_end_conv2D1x1_units=0,
        end_conv2D1x1_L2reg=False,
        end_conv2D1x1_DP_rate=0,
        use_GMP=False,
        use_DenseV0=False,
        Dense1_DP_rate=0,
        Dense2_DP_rate=0):
    model_name = "pretrained_VGG16BNConv"

    VGG16_conv_blocks = Sequential([
        layers.Input(shape=input_shape),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(256, (3, 3), padding='same', ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(256, (3, 3), padding='same', ),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(512, (3, 3), padding='same', ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(512, (3, 3), padding='same', ),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(512, (3, 3), padding='same', ),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(512, (3, 3), padding='same', ),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    ])
    model = VGG16_conv_blocks

    # 遷移 pretrained layer weights
    conv_layer_weight_list = get_conv_weights_from_pretrained_VGG16()
    if input_shape[2] == 3:
        model = set_conv_weights_to_VGG16_conv_layers(
            model, conv_layer_weight_list, channel=3)
    # if input_shape[2] == 4:
    #     model = set_conv_weights_to_VGG16_conv_layers(
    #         model, conv_layer_weight_list, channel=4)


    # 設定網路權重可否訓練
    index = 0
    for layer in model.layers:
        index += 1
        if "batch_normalization" in layer.name:  # BN layer權重為可訓練
            layer.trainable = True
        else:  # 根據conv_blocks_trainable決定BN layer以外的網路權重可否訓練
            if conv_blocks_trainable is False:
                layer.trainable = False
            else:
                layer.trainable = True
        logger.info("layer_index:{} , layer.name:{}  , layer.trainable:{}".format(
            index, layer.name, layer.trainable))  # 預設皆為True

    # 在骨幹網路後接上1x1 Conv2d layer
    L2regularizer = tf.keras.regularizers.L2(l2=0.01)  # default value: 0.01
    if ConvBlocks_end_conv2D1x1_units != 0:
        if end_conv2D1x1_DP_rate != 0:
            model.add(Dropout(end_conv2D1x1_DP_rate))

        if end_conv2D1x1_L2reg is True:
            end_conv2D = Conv2D(ConvBlocks_end_conv2D1x1_units, (1, 1),
                                padding='valid', kernel_regularizer=L2regularizer)
        else:
            end_conv2D = Conv2D(
                ConvBlocks_end_conv2D1x1_units, (1, 1), padding='valid')

        logger.info("endConv2d.trainable: %s ", end_conv2D.trainable)
        model.add(end_conv2D)

    if use_DenseV0 is True:
        model.add(Flatten())

        if Dense1_DP_rate != 0:
            model.add(Dropout(Dense1_DP_rate))

        model.add(
            layers.Dense(
                4096, activation='relu',
                kernel_initializer=tf.keras.initializers.HeUniform(),
                bias_initializer=tf.keras.initializers.HeUniform()))

        if Dense2_DP_rate != 0:
            model.add(Dropout(Dense2_DP_rate))

        model.add(
            layers.Dense(
                4096, activation='relu',
                kernel_initializer=tf.keras.initializers.HeUniform(),
                bias_initializer=tf.keras.initializers.HeUniform()))
    elif use_GMP is True:
        model.add(GlobalMaxPooling2D())
    else:
        model.add(GlobalAveragePooling2D())

    if DP_rate != 0:
        model.add(Dropout(DP_rate))

    if L2reg is False:
        model.add(layers.Dense(2, activation='softmax', kernel_initializer=tf.keras.initializers.HeUniform(
        ), bias_initializer=tf.keras.initializers.HeUniform()))
    else:
        model.add(layers.Dense(2, activation='softmax', kernel_regularizer=L2regularizer,
                               kernel_initializer=tf.keras.initializers.HeUniform(),
                               bias_initializer=tf.keras.initializers.HeUniform()))

    model = tf.keras.Model(inputs=model.input, outputs=[model.output])

    return model, model_name
