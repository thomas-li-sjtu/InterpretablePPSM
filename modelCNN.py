import os
from functools import partial
import tensorflow as tf
import numpy as np
import architectureCNN as architecture

PAD = 0


# -----------------------------------------------------------------------------------------------
def make_model(hparams, DICT_SIZE, MAX_LEN):
    """
    建立模型
    Args:
        hparams: 超参数
        DICT_SIZE: 口令集字符的字典
        MAX_LEN: 口令最长长度

    Returns: model实例
    """
    arch_id = hparams['arch']
    arch = architecture.archMap(arch_id)  # 根据超参数'arch'，确定网络结构

    input = tf.keras.layers.Input(MAX_LEN, dtype=tf.int32)  # 输入层
    logits, z, att_ws = arch(input, MAX_LEN, DICT_SIZE, hparams)  # z和att_ws是相同的，z为encode与decode中间层输出

    p = tf.nn.softmax(logits, 2)
    prediction = tf.argmax(p, 2, output_type=tf.int32)

    # TODO 为什么outputs可以是一个列表？
    model = tf.keras.Model(inputs=input, outputs=[logits, p, prediction, z, att_ws])  # inputs与outputs一定是Layer调用输出的张量

    return model


# -----------------------------------------------------------------------------------------------
def loss_function(real, logits, z, hparams):
    batch_size = hparams['batch_size']
    alpha = hparams['alpha']
    loss_type = hparams['loss_type']

    # if loss_type == 0:
    loss_class = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(real, logits)

    shape = batch_size, z.shape.as_list()[1]

    ztarget = tf.random.normal(shape)
    latent_reg = mmd_loss(z, ztarget) * alpha

    loss_ = loss_class + latent_reg

    return loss_


# -----------------------------------------------------------------------------------------------

def make_train_predict(hparams, optimizer, DICT_SIZE, MAX_LEN):
    """
    模型建立
    Args:
        hparams: 超参数
        optimizer: 优化器
        DICT_SIZE: 字典长度（vocal_size）
        MAX_LEN: 口令最大长度

    Returns: model与训练函数、预测函数
    """
    model = make_model(hparams, DICT_SIZE, MAX_LEN)

    print(model.summary())

    @tf.function
    def train_step(data):
        features, prediction_mask, labels = data
        with tf.GradientTape() as tape:
            # forward
            logits, p, prediction, z, _ = model(features, training=True)
            loss = loss_function(labels, logits, z, hparams)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 迭代

        return loss, p, prediction

    @tf.function
    def predict_step(data):
        features, prediction_mask, labels = data
        # forward
        logits, p, prediction, z, _ = model(features, training=False)
        loss = loss_function(labels, logits, z, hparams)

        return loss, p, prediction

    return model, train_step, predict_step


# ----------______________-----------------________________-----------------

sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]


def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel):
    with tf.name_scope('MaximumMeanDiscrepancy'):
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def mmd_loss(source_samples, target_samples, scope=None):
    """ from https://github.com/tensorflow/models/blob/master/research/domain_adaptation/domain_separation/losses.py """

    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value)

    return loss_value
