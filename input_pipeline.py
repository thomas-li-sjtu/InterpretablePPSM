import numpy as np
import tensorflow as tf
import os
import myPickle

XNAME = 'X.txt'
CMNAME = 'charmap.pickle'
ENCODING = 'ascii'
buffer_size = 10000

# Special tokens
MASK = 3
END = 2
PAD = 0
################

MAX_MASKED = .3


# def mask(x, xl):
#    if INCLUDE_END_SYMBOL: xl -= 1
#    k = np.random.randint(0, xl)
#    y = x[k]
#    x[k] = MASK
#    return x, [y], [k]

def mask(x_index_copy, x_len, MAX_MASKED, INCLUDE_END_SYMBOL):
    if INCLUDE_END_SYMBOL:
        x_len -= 1

    if MAX_MASKED == -1:
        # single missing
        masked_index = [np.random.randint(0, x_len)]
    else:
        # 随机选择mask的下标
        num_masked = int(x_len * MAX_MASKED)
        removed = []
        masked_index = np.random.randint(0, x_len, size=num_masked)

    # 将kk中的下标mask
    for k in masked_index:
        x_index_copy[k] = MASK

    return x_index_copy, [], masked_index


def idx2string(P, CM_):
    return ''.join([CM_[p] for p in P if p > 0 and p != END])


def string2idx(x, CM, MAX_LEN, CMm, INCLUDE_END_SYMBOL):
    f = lambda x: CM[x] if x in CM else CMm  # 匿名函数，map函数通过此匿名函数的charmap将字符映射为数字
    x = list(map(f, x))
    if INCLUDE_END_SYMBOL:
        x += [END]  # 增加结尾符
    x += [0] * (MAX_LEN - len(x))  # 将每个口令补齐到相同长度
    return np.array(x)


def makeIterInput(home, batch_size, MAX_MASKED, INCLUDE_END_SYMBOL, MAX_LEN=32, buffer_size=buffer_size,
                  for_prediction=False):
    XPATH = os.path.join(home, XNAME)  # 数据集路径

    CMPATH = os.path.join(home, CMNAME)
    CM = myPickle.load(CMPATH)  # 导入charmap
    vocab_size = max(CM.values()) + 1  # 字典大小

    def G(*args):
        """
        生成器
        Args:
            *args:
        Returns:
        """
        # for each chunk
        with open(XPATH, encoding=ENCODING, errors='ignore') as f:
            for x in f:
                x = x[:-1]  # 去除最后的换行符
                x_len = len(x)

                # if not INCLUDE_END_SYMBOL: print("NO <END>")

                if x_len > MAX_LEN - int(INCLUDE_END_SYMBOL):  # 需要留下INCLUDE_END_SYMBOL长度的字符作为口令结尾
                    # 口令过长
                    continue

                x_index = string2idx(x, CM, MAX_LEN, vocab_size, INCLUDE_END_SYMBOL)  # 口令在charmap下的对应下标列表，长度补全

                # .copy()复制新的列表——列表第一层为深拷贝
                # 随机选择部分下标将其遮盖，返回遮盖后的结果和遮盖的下标
                x_index_in, _, masked_index = mask(x_index.copy(), x_len, MAX_MASKED, INCLUDE_END_SYMBOL)

                prediction_mask = np.zeros(MAX_LEN, np.int32)  # 对遮盖后的结果取反，获得类似[0,0,0,1,0,0]的结果
                for k in masked_index:
                    prediction_mask[k] = 1

                xi_out = x_index

                yield x_index_in, prediction_mask, xi_out

    # 构造数据集（以管道的方式输入网络训练）  输入生成器，输出类型和输出shape，None表明shape自动生成；输出的元组由生成器的输出决定
    dataset = tf.data.Dataset.from_generator(G, (tf.int32, tf.int32, tf.int32), ((None,), (None,), (None,)))

    if not for_prediction:
        # 以 batch_size 打乱数据集
        dataset = dataset.shuffle(buffer_size)
    # padded_shapes = (
    #     tf.TensorShape([None]),
    #     tf.TensorShape([None]),
    #     tf.TensorShape([None])
    # )
    # dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True)

    # 将此数据集的多个连续元素 (可能具有不同的形状) 合并到单个元素中。结果元素中的张量有一个额外的外部维度, 并填充到 padded_shapes 中的相应形状
    dataset = dataset.padded_batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=buffer_size)  # 预取数据,将生成数据的时间和使用数据的时间分离,在请求元素之前从输入数据集中预取这些元素

    return dataset, vocab_size + 1, CM
