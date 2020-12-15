import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from architectureCNN import *
from input_pipeline import PAD


def archMap(arch_id):
    """
    返回模型结构函数
    Args:
        arch_id: 超参数arch

    Returns: 模型结构函数
    """
    m = {
        1: resnet_classic,  # 应当将ORIGINAL.gin的arch改为1，或将这里的1改为0
    }  # 模型函数字典
    return m[arch_id]


EPSILON_NORM = 1e-6
layer_norm = lambda x: tf.keras.layers.LayerNormalization(epsilon=EPSILON_NORM)(x)


def ResBlockDeepBNK(inputs, dim, filter_size, activation, with_batch_norm=True, training=True):
    x = inputs

    dim_BNK = dim // 2

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv1D(dim_BNK, 3, padding='same')(x)

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv1D(dim_BNK, filter_size, padding='same')(x)

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv1D(dim, 3, padding='same')(x)

    return inputs + (0.3 * x)


def ResBlockDeepBNK_separable(inputs, dim, filter_size, activation, with_batch_norm=True, training=True):
    x = inputs

    dim_BNK = dim // 2

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.SeparableConv1D(dim_BNK, 3, padding='same')(x)

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.SeparableConv1D(dim_BNK, filter_size, padding='same')(x)

    if with_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.SeparableConv1D(dim, 3, padding='same')(x)

    return inputs + (0.3 * x)


BLOCKS_MAP = {
    0: ResBlockDeepBNK,
    1: ResBlockDeepBNK_separable
}


def resnet_backbone(x, max_len, vocab_size, hparams):
    """
    定义网络模型
    Args:
        x: 输入层
        max_len: 口令最长长度
        vocab_size: 字典长度
        hparams: 超参数

    Returns: decode的结果与中间层输出
    """
    latent_size = hparams['latent_size']  # 128
    layer_dim = hparams['k']  # 128
    embedding_size = hparams['embedding_size']
    filter_size = hparams['filter_size']  # 3
    n_blocks = hparams['n_blocks']  # 10
    block_type = hparams['block_type']
    activation = hparams['activation']

    batch_norm = True

    block = BLOCKS_MAP[block_type]

    x = tf.keras.layers.Embedding(vocab_size, embedding_size, name="char_embedding")(x)
    # x = tf.one_hot(x, vocab_size)

    output_shape = max_len, vocab_size

    x = tf.keras.layers.Conv1D(layer_dim, filter_size, padding='same')(x)

    # encoder
    for i in range(n_blocks):
        x = block(x, layer_dim, activation=activation, with_batch_norm=batch_norm, filter_size=filter_size)

    # latent space
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(latent_size)(x)
    x = tf.keras.layers.Dense(output_shape[0] * layer_dim)(z)
    x = tf.keras.layers.Reshape([output_shape[0], layer_dim])(x)

    # decoder
    for i in range(n_blocks):
        x = block(x, layer_dim, activation=activation, with_batch_norm=batch_norm, filter_size=filter_size)

    return x, z


def resnet_classic(input, max_len, vocab_size, hparams):
    """

    Args:
        input: 输入层
        max_len: 口令最长长度
        vocab_size: 数据集字符字典长度
        hparams: 超参数字典

    Returns:
    """
    output_shape = max_len, vocab_size
    input, z = resnet_backbone(input, max_len, vocab_size, hparams)

    input = tf.keras.layers.Flatten()(input)  # 展平
    input = tf.keras.layers.Dense(output_shape[0] * output_shape[1])(input)
    logits = tf.keras.layers.Reshape((output_shape[0], output_shape[1]))(input)

    return logits, z, z
