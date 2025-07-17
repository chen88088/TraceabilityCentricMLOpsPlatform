# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models
import logging
import numpy as np

from skimage.transform import resize
from tensorflow.keras.models import load_model

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
import glob
import shutil
import openpyxl
import datetime
from skimage.measure import label, regionprops

from tensorflow import float32

# from keras.models 改成 from tensorflow.keras.models !!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D
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


def pretrained_VGG16Conv_GAP(
        input_shape=(500, 450, 3),
        DP_rate=0,
        conv_blocks_trainable=False,
        use_GMP=False,
        use_DenseV0=False,
        Dense1_DP_rate=0,
        Dense2_DP_rate=0):
    model_name = "pretrained_VGG16Conv"

    VGG16_conv_blocks = Sequential([
        layers.Input(shape=input_shape),
        Conv2D(64, (3, 3), padding='same'),
        Activation("relu"),
        Conv2D(64, (3, 3), padding='same'),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), padding='same'),
        Activation("relu"),
        Conv2D(128, (3, 3), padding='same'),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', ),
        Activation("relu"),
        Conv2D(256, (3, 3), padding='same', ),
        Activation("relu"),
        Conv2D(256, (3, 3), padding='same', ),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', ),
        Activation("relu"),
        Conv2D(512, (3, 3), padding='same', ),
        Activation("relu"),
        Conv2D(512, (3, 3), padding='same', ),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', ),
        Activation("relu"),
        Conv2D(512, (3, 3), padding='same', ),
        Activation("relu"),
        Conv2D(512, (3, 3), padding='same', ),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    ])
    model = VGG16_conv_blocks

    # 遷移 pretrained layer weights
    conv_layer_weight_list = get_conv_weights_from_pretrained_VGG16()
    if input_shape[2] == 3:
        model = set_conv_weights_to_VGG16_conv_layers(
            model, conv_layer_weight_list, channel=3)
    if input_shape[2] == 4:
        model = set_conv_weights_to_VGG16_conv_layers(
            model, conv_layer_weight_list, channel=4)

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

    model.add(layers.Dense(2, kernel_initializer=tf.keras.initializers.HeUniform(
    ), bias_initializer=tf.keras.initializers.HeUniform()))
    model.add(Activation("softmax"))

    model = tf.keras.Model(inputs=model.input, outputs=[model.output])

    return model, model_name

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
    if input_shape[2] == 4:
        model = set_conv_weights_to_VGG16_conv_layers(
            model, conv_layer_weight_list, channel=4)

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


def pretrained_VGG16ConvX_add_preprocess_input(
        model,
        model_name,
        input_shape,
        split_up_fourth_channel=False
):
    model_name = model_name + "_PI"
    main_model = model
    if split_up_fourth_channel is False:
        i = tf.keras.layers.Input(input_shape, dtype=tf.uint8)
        x = tf.cast(i, tf.float32)
        preprocess_input_output = tf.keras.applications.vgg16.preprocess_input(
            x)

        output = main_model(preprocess_input_output)
        model = tf.keras.Model(inputs=[i], outputs=[output])
    elif split_up_fourth_channel is True:
        nirrg_input = layers.Input(shape=(input_shape[0], input_shape[1], 3))
        preprocess_input_output = tf.keras.applications.vgg16.preprocess_input(
            nirrg_input)

        mask_input = layers.Input(shape=(input_shape[0], input_shape[1], 1))

        stacked_four_channel = tf.concat(
            [preprocess_input_output, mask_input], 3)
        output = main_model(stacked_four_channel)
        model = tf.keras.Model(
            inputs=[nirrg_input, mask_input], outputs=[output])

    model.summary()

    return model, model_name


# -


# +
# ResNet50V2
# +

def base_model_layer_trainable_setting(
        base_model, conv_blocks_trainable=False, top_BN_trainable=False):
    for layer in base_model.layers:

        if conv_blocks_trainable is False:
            layer.trainable = False
        else:
            layer.trainable = True

        # # 針對BN layer再設定
        # if "bn" in layer.name:
        #     if top_BN_trainable == False:
        #         layer.trainable = False
        #     else:
        #         logger.info("*** top_BN_trainable == True")
        #         layer.trainable = True

    return base_model


def set_pretrained_weights_to_four_channel_model(
        pretrained_Conv_blocks,
        four_channel_Conv_blocks,
        set_first_conv_layer_trainable=False):
    layer_count = -1
    conv_count = 0
    for pretrained_layer, new_layer in zip(
            pretrained_Conv_blocks.layers, four_channel_Conv_blocks.layers):
        layer_count += 1
        # logger.info(pretrained_layer.name, new_layer.name)
        pretrained_layer_type = str(type(pretrained_layer))
        if "Conv2D" in pretrained_layer_type:
            conv_count += 1

        if "KerasTensor" in pretrained_layer_type:  # input layer or preprocess_input
            pass
        else:
            if conv_count == 1:  # input的通道數只影響接在input layer後面的第一層
                conv_count = 999  # 初始化 再也不會進來此
                if pretrained_layer.weights != []:
                    # logger.info(len(pretrained_layer.weights))
                    # logger.info(pretrained_layer.weights[0].shape, new_layer.weights[0].shape)

                    kernel_weight = pretrained_layer.weights[0]
                    n_channel = kernel_weight.shape[2]

                    four_channel_kernel_weight = np.zeros(
                        (kernel_weight.shape[0], kernel_weight.shape[1], n_channel + 1, kernel_weight.shape[3]))
                    four_channel_bias_weight = np.zeros((n_channel + 1))

                    # Input layer後的第一個pretrained layer只有3通道weight,
                    # 取出Keras預設初始化的權重
                    initialized_new_layer_weight = new_layer.weights[0][:, :, 3, :]
                    initialized_new_layer_weight = tf.expand_dims(initialized_new_layer_weight, axis=2)

                    # 放入pretrained layer的3通道weight
                    four_channel_kernel_weight[:, :, 0:n_channel, :] = kernel_weight[:, :, 0:n_channel, :]
                    # 放入Keras預設初始化的權重
                    four_channel_kernel_weight[:, :, 3, :] = initialized_new_layer_weight[:, :, 0, :]
                    # logger.info("four_channel_kernel_weight[:,:,0,:]:",four_channel_kernel_weight[:,:,0,:])
                    # 放入pretrained layer的3通道weight
                    four_channel_kernel_weight[:, :, 0, :] = kernel_weight[:, :, 0, :]
                    # 放入pretrained layer的3通道weight
                    four_channel_kernel_weight[:, :, 1, :] = kernel_weight[:, :, 1, :]
                    # 放入pretrained layer的3通道weight
                    four_channel_kernel_weight[:, :, 2, :] = kernel_weight[:, :, 2, :]
                    # logger.info("four_channel_kernel_weight[:,:,0,:]:",four_channel_kernel_weight[:,:,0,:])

                    #                     logger.info("type(four_channel_kernel_weight)",type(four_channel_kernel_weight))
                    #                     logger.info("four_channel_kernel_weight", four_channel_kernel_weight)

                    if len(pretrained_layer.weights) == 1:
                        four_channel_weights = [four_channel_kernel_weight]
                    if len(pretrained_layer.weights) == 2:
                        bias_weight = pretrained_layer.weights[1]
                        four_channel_bias_weight = bias_weight  # type: tensor

                        # #type: tensor =>numpy array
                        four_channel_bias_weight = np.array(four_channel_bias_weight)

                        # 同為numpy array
                        four_channel_weights = [four_channel_kernel_weight, four_channel_bias_weight]

                    new_layer.set_weights(four_channel_weights)

                    if set_first_conv_layer_trainable is True:
                        new_layer.trainable = True

            else:
                if pretrained_layer.weights != []:
                    # logger.info(pretrained_layer.weights[0].shape, new_layer.weights[0].shape)
                    new_layer.set_weights(pretrained_layer.get_weights())

    four_channel_base_model_with_pretrained_weights = tf.keras.Model(
        inputs=[four_channel_Conv_blocks.input], outputs=[four_channel_Conv_blocks.output])
    return four_channel_base_model_with_pretrained_weights


def pretrained_ResNet50V2Conv_GAP(
        input_shape=(256, 256, 3),
        conv_blocks_trainable=False,
        set_first_conv_layer_trainable=False,
        Dense_DP_rate=0,
        Dense_L2reg=False,
        Dense_max_norm_value=None,
        end_conv2D1x1_count=0,
        end_conv2D1x1_units=0,
        end_conv2D1x1_BN=False,
        end_conv2D1x1_DP_rate=0,
        end_conv2D1x1_L2reg=False,
        end_conv2D1x1_max_norm_value=None,
        use_GMP=False,
        use_Dense=False):
    """
    原始設計用GAP+Dense(1000)分類

    top :
        GAP
        Dense(1000)
    """
    model_name = "pretrained_ResNet50V2Conv"

    if input_shape[-1] == 3:
        base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=None,
                                                      input_shape=input_shape)
        base_model = base_model_layer_trainable_setting(
            base_model, conv_blocks_trainable)
    elif input_shape[-1] == 4:
        pretrained_Conv_blocks = tf.keras.applications.ResNet50V2(
            weights='imagenet', include_top=False)
        four_channel_Conv_blocks = tf.keras.applications.ResNet50V2(
            input_shape=input_shape, weights=None, include_top=False)
        four_channel_base_model_with_pretrained_weights = set_pretrained_weights_to_four_channel_model(
            pretrained_Conv_blocks,
            four_channel_Conv_blocks,
            set_first_conv_layer_trainable)
        base_model = four_channel_base_model_with_pretrained_weights

    ConvBlocks_output = base_model.output

    L2regularizer = tf.keras.regularizers.L2(l2=0.01)

    He_initializer = tf.keras.initializers.HeUniform()

    if end_conv2D1x1_count != 0:
        if end_conv2D1x1_L2reg is False:
            end_conv2D1x1_kernel_regularizer = None
        else:
            end_conv2D1x1_kernel_regularizer = L2regularizer

        if end_conv2D1x1_max_norm_value is None:
            end_conv2D1x1_kernel_constraint = None
        else:
            end_conv2D1x1_kernel_constraint = tf.keras.constraints.MaxNorm(
                max_value=end_conv2D1x1_max_norm_value, axis=0)


        for i in range(0, end_conv2D1x1_count):

            if end_conv2D1x1_DP_rate != 0:
                ConvBlocks_output = Dropout(
                    end_conv2D1x1_DP_rate)(ConvBlocks_output)
                model_name = model_name + "_DP{}".format(str(end_conv2D1x1_DP_rate))

            #             zeroPadding2D_layer = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
            end_conv2D = Conv2D(end_conv2D1x1_units, (1, 1),
                                kernel_regularizer=end_conv2D1x1_kernel_regularizer,
                                kernel_constraint=end_conv2D1x1_kernel_constraint,
                                kernel_initializer=He_initializer, bias_initializer=He_initializer)
            logger.info("EndConv2d.trainable:", end_conv2D.trainable)
            bn_layer = BatchNormalization()
            activation_layer = Activation('relu')

            #             ConvBlocks_output = zeroPadding2D_layer(ConvBlocks_output)
            ConvBlocks_output = end_conv2D(ConvBlocks_output)
            if end_conv2D1x1_BN is True:
                ConvBlocks_output = bn_layer(ConvBlocks_output)
                logger.info("EndConv2d with BN")
            else:
                logger.info("EndConv2d no BN")
            ConvBlocks_output = activation_layer(ConvBlocks_output)

    if use_Dense is True:
        gap = Flatten()(ConvBlocks_output)
        gap = layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(
        ), bias_initializer=tf.keras.initializers.HeUniform())(gap)
        gap = layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(
        ), bias_initializer=tf.keras.initializers.HeUniform())(gap)
    elif use_GMP is True:
        gap = GlobalMaxPooling2D()(ConvBlocks_output)
    else:
        gap = GlobalAveragePooling2D()(ConvBlocks_output)

    if Dense_L2reg is False:
        Dense_kernel_regularizer = None
    else:
        Dense_kernel_regularizer = L2regularizer

    if Dense_max_norm_value is None:
        Dense_kernel_constraint = None
    else:
        Dense_kernel_constraint = tf.keras.constraints.MaxNorm(
            max_value=Dense_max_norm_value, axis=0)

    if Dense_DP_rate != 0:
        gap = Dropout(Dense_DP_rate)(gap)

    if Dense_L2reg is False:
        if Dense_max_norm_value is None:
            model_output = Dense(
                2, activation='softmax', kernel_initializer=He_initializer, bias_initializer=He_initializer)(gap)
        if Dense_max_norm_value is not None:
            model_output = Dense(2, activation='softmax', kernel_constraint=Dense_kernel_constraint,
                                 kernel_initializer=He_initializer, bias_initializer=He_initializer)(gap)
    else:
        model_output = Dense(2, activation='softmax', kernel_regularizer=L2regularizer,
                             kernel_initializer=He_initializer, bias_initializer=He_initializer)(gap)

    model = tf.keras.Model(inputs=base_model.input, outputs=[model_output])

    # model.summary()
    return model, model_name


def pretrained_ResNet50V2Conv_GAP_Dense(
        input_shape=(256, 256, 3),
        conv_blocks_trainable=False,
        set_first_conv_layer_trainable=False,
        new_Dense_units=0,
        new_Dense_DP_rate=0,
        new_Dense_L2reg=False,
        new_Dense_max_norm_value=0,
        Dense_DP_rate=0,
        Dense_L2reg=False,
        Dense_use_max_norm_reg=False,
        end_conv2D1x1_count=0,
        end_conv2D1x1_units=0,
        end_conv2D1x1_DP_rate=0,
        end_conv2D1x1_L2reg=False,
        end_conv2D1x1_max_norm_value=0,
        use_GMP=False):
    """
    原始設計用GAP+Dense(1000)分類

    top :
        GAP
        Dense(1000)
    """
    model_name = "pretrained_ResNet50V2Conv"
    if input_shape[-1] == 3:
        base_model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=None,
                                                      input_shape=input_shape)
        base_model = base_model_layer_trainable_setting(
            base_model, conv_blocks_trainable)
    elif input_shape[-1] == 4:
        pretrained_Conv_blocks = tf.keras.applications.ResNet50V2(
            weights='imagenet', include_top=False)
        four_channel_Conv_blocks = tf.keras.applications.ResNet50V2(
            input_shape=input_shape, weights=None, include_top=False)
        four_channel_base_model_with_pretrained_weights = set_pretrained_weights_to_four_channel_model(
            pretrained_Conv_blocks,
            four_channel_Conv_blocks,
            set_first_conv_layer_trainable)
        base_model = four_channel_base_model_with_pretrained_weights

    ConvBlocks_output = base_model.output

    L2regularizer = tf.keras.regularizers.L2(l2=0.01)
    if end_conv2D1x1_max_norm_value != 0:
        max_norm = tf.keras.constraints.MaxNorm(
            max_value=end_conv2D1x1_max_norm_value, axis=0)
    He_initializer = tf.keras.initializers.HeUniform()

    if end_conv2D1x1_count != 0:
        if end_conv2D1x1_L2reg is False:
            end_conv2D1x1_kernel_regularizer = None
        else:
            end_conv2D1x1_kernel_regularizer = L2regularizer

        if end_conv2D1x1_max_norm_value == 0:
            end_conv2D1x1_kernel_constraint = None
        else:
            end_conv2D1x1_kernel_constraint = max_norm

        for i in range(0, end_conv2D1x1_count):

            if end_conv2D1x1_DP_rate != 0:
                ConvBlocks_output = Dropout(
                    end_conv2D1x1_DP_rate)(ConvBlocks_output)

            zeroPadding2D_layer = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
            end_conv2D = Conv2D(end_conv2D1x1_units, (1, 1),
                                kernel_regularizer=end_conv2D1x1_kernel_regularizer,
                                kernel_constraint=end_conv2D1x1_kernel_constraint,
                                kernel_initializer=He_initializer, bias_initializer=He_initializer)
            logger.info("EndConv2d.trainable:", end_conv2D.trainable)
            bn_layer = BatchNormalization()
            activation_layer = Activation('relu')

            ConvBlocks_output = zeroPadding2D_layer(ConvBlocks_output)
            ConvBlocks_output = end_conv2D(ConvBlocks_output)
            ConvBlocks_output = bn_layer(ConvBlocks_output)
            logger.info("EndConv2d + BN")
            ConvBlocks_output = activation_layer(ConvBlocks_output)

    if use_GMP is False:
        gap = GlobalAveragePooling2D()(ConvBlocks_output)
    else:
        gap = GlobalMaxPooling2D()(ConvBlocks_output)

    if Dense_L2reg is False:
        Dense_kernel_regularizer = None
    else:
        Dense_kernel_regularizer = L2regularizer

    if Dense_use_max_norm_reg is False:
        Dense_kernel_constraint = None
    else:
        Dense_kernel_constraint = max_norm

    if Dense_DP_rate != 0:
        gap = Dropout(Dense_DP_rate)(gap)

    if new_Dense_units != 0:
        if new_Dense_L2reg is False:
            new_Dense_kernel_regularizer = None
        else:
            new_Dense_kernel_regularizer = L2regularizer

        if new_Dense_max_norm_value == 0:
            new_Dense_kernel_constraint = None
        else:
            new_Dense_kernel_constraint = max_norm

        for i in range(0, new_Dense_units):
            if new_Dense_DP_rate != 0:
                gap = Dropout(new_Dense_DP_rate)(gap)

            new_Dense = Dense(new_Dense_units,
                              kernel_regularizer=new_Dense_kernel_regularizer,
                              kernel_constraint=new_Dense_kernel_constraint,
                              kernel_initializer=He_initializer, bias_initializer=He_initializer)
            logger.info("new_Dense.trainable:", new_Dense.trainable)
            bn_layer = BatchNormalization()
            activation_layer = Activation('relu')

            gap = new_Dense(gap)
            gap = bn_layer(gap)
            logger.info("new_Dense + BN")
            gap = activation_layer(gap)

    if Dense_L2reg is False:
        if Dense_use_max_norm_reg is False:
            model_output = Dense(
                2, activation='softmax', kernel_initializer=He_initializer, bias_initializer=He_initializer)(gap)
        if Dense_use_max_norm_reg is True:
            model_output = Dense(2, activation='softmax', kernel_constraint=max_norm,
                                 kernel_initializer=He_initializer, bias_initializer=He_initializer)(gap)
    else:
        model_output = Dense(2, activation='softmax', kernel_regularizer=L2regularizer,
                             kernel_initializer=He_initializer, bias_initializer=He_initializer)(gap)

    model = tf.keras.Model(inputs=base_model.input, outputs=[model_output])

    return model, model_name


def pretrained_ResNet50V2ConvX_add_preprocess_input(
        model,
        model_name,
        input_shape=(256, 256, 3),
        optimizer_learning_rate=0.001,
        split_up_fourth_channel=False,
        loaded_trained_model_path=None
):
    model_name = model_name + "_PI"
    main_model = model

    if loaded_trained_model_path is not None:
        main_model = tf.keras.Model(
            inputs=[main_model.input], outputs=[main_model.output])
        main_model.load_weights(loaded_trained_model_path)

    if split_up_fourth_channel is False:
        i = tf.keras.layers.Input(input_shape, dtype=tf.uint8)
        x = tf.cast(i, tf.float32)
        preprocess_input_output = tf.keras.applications.resnet_v2.preprocess_input(
            x)

        output = main_model(preprocess_input_output)
        model = tf.keras.Model(inputs=[i], outputs=[output])
    elif split_up_fourth_channel is True:
        nirrg_input = layers.Input(shape=(input_shape[0], input_shape[1], 3))
        preprocess_input_output = tf.keras.applications.resnet_v2.preprocess_input(
            nirrg_input)

        mask_input = layers.Input(shape=(input_shape[0], input_shape[1], 1))

        stacked_four_channel = tf.concat(
            [preprocess_input_output, mask_input], 3)
        output = main_model(stacked_four_channel)
        model = tf.keras.Model(
            inputs=[nirrg_input, mask_input], outputs=[output])

    #     model.summary()

    return model, model_name


# -


# +
def pretrained_InceptionV3ConvX_add_preprocess_input(
        model,
        model_name,
        input_shape=(256, 256, 3),
        split_up_fourth_channel=False
):
    model_name = model_name + "_PI"
    main_model = model
    if split_up_fourth_channel is False:
        i = tf.keras.layers.Input(input_shape, dtype=tf.uint8)
        x = tf.cast(i, tf.float32)
        preprocess_input_output = tf.keras.applications.inception_v3.preprocess_input(
            x)

        output = main_model(preprocess_input_output)
        model = tf.keras.Model(inputs=[i], outputs=[output])
    elif split_up_fourth_channel is True:
        nirrg_input = layers.Input(shape=(input_shape[0], input_shape[1], 3))
        preprocess_input_output = tf.keras.applications.inception_v3.preprocess_input(
            nirrg_input)

        mask_input = layers.Input(shape=(input_shape[0], input_shape[1], 1))

        stacked_four_channel = tf.concat(
            [preprocess_input_output, mask_input], 3)
        output = main_model(stacked_four_channel)
        model = tf.keras.Model(
            inputs=[nirrg_input, mask_input], outputs=[output])

    #     model.summary()

    return model, model_name


def pretrained_InceptionV3Conv_GAP(
        input_shape=(256, 256, 3),
        optimizer_learning_rate=0.001,
        ConvBlocks_BN=False,
        Dense_DP_rate=0,
        conv_blocks_trainable=False,
        top_BN_trainable=False,
        set_first_conv_layer_trainable=False,
        Dense_L2reg=False,
        use_GMP=False):
    """
    原始設計用GAP+Dense(1000)分類

    top :
        GAP
        Dense(1000)
    """
    He_initializer = tf.keras.initializers.HeUniform()

    model_name = "pretrained_InceptionV3Conv"

    if input_shape[-1] == 3:
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
                                                       input_shape=input_shape)

        base_model = base_model_layer_trainable_setting(
            base_model, conv_blocks_trainable, top_BN_trainable)
    elif input_shape[-1] == 4:
        pretrained_Conv_blocks = tf.keras.applications.InceptionV3(
            weights='imagenet', include_top=False)
        four_channel_Conv_blocks = tf.keras.applications.InceptionV3(
            input_shape=input_shape, weights=None, include_top=False)
        four_channel_base_model_with_pretrained_weights = set_pretrained_weights_to_four_channel_model(
            pretrained_Conv_blocks,
            four_channel_Conv_blocks,
            set_first_conv_layer_trainable)
        base_model = four_channel_base_model_with_pretrained_weights

    ConvBlocks_output = base_model.output
    if ConvBlocks_BN is True:
        ConvBlocks_output = BatchNormalization(
            name="ConvBlocks_bn")(ConvBlocks_output)

    if use_GMP is False:
        gap = GlobalAveragePooling2D()(ConvBlocks_output)
    else:
        gap = GlobalMaxPooling2D()(ConvBlocks_output)

    if Dense_DP_rate != 0:
        gap = Dropout(Dense_DP_rate)(gap)

    if Dense_L2reg is False:
        model_output = Dense(
            2, activation='softmax', kernel_initializer=He_initializer, bias_initializer=He_initializer)(gap)
    else:
        L2reg = tf.keras.regularizers.L2(l2=0.01)
        model_output = Dense(2, activation='softmax', kernel_regularizer=L2reg,
                             kernel_initializer=He_initializer, bias_initializer=He_initializer)(gap)

    model = tf.keras.Model(inputs=base_model.input, outputs=[model_output])

    #     model.summary()
    return model, model_name

# -
