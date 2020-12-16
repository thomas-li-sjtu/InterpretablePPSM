# pip install h5py==2.10
# 否则报错：
#   model_config = json.loads(model_config.decode('utf-8'))
# AttributeError: 'str' object has no attribute 'decode'
BATCH_SIZE = 1024
MAX_LEN = 16
CHARMAP = './charmap.pickle'
TERMINAL_SYMBOL = False
ENCODING = 'ascii'
# ---------------------------

from inference import Inference
import myPickle, sys
import tensorflow as tf


def read_passwords(path, encoding=ENCODING, MIN_LEN=0, MAX_LEN=16):
    """
    从文件读入口令
    Args:
        path: 文件路径
        encoding: 编码
        MIN_LEN: 口令最短长度
        MAX_LEN: 口令最长长度

    Returns: 口令列表
    """
    X = []
    with open(path, encoding=encoding, errors='ignore') as f:
        for x in f:
            x = x[:-1]  # 除去换行符
            if MAX_LEN >= len(x) >= MIN_LEN:
                X.append(x)
    return X


def write_tsv(output, passwords, log_probability, encoding=ENCODING):
    """
    输出结果到文件中
    Args:
        output: 输出路径
        passwords: 口令列表
        log_probability: 非归一化的概率列表(log)
        encoding: 编码

    Returns:
    """
    assert len(passwords) == len(log_probability)
    n = len(passwords)
    with open(output, 'w', encoding=encoding) as f:
        for x, p in zip(passwords, log_probability):
            print("%s\t%f" % (x, p), file=f)


if __name__ == '__main__':
    try:
        model_path = sys.argv[1]
        password_file = sys.argv[2]
        output_path = sys.argv[3]
    except:
        print("USAGE: model_path.h5 password_path.txt output_path.txt")
        sys.exit(1)

    passwords = read_passwords(password_file)
    charmap = myPickle.load(CHARMAP)

    model = tf.keras.models.load_model(model_path, compile=False)
    infer = Inference(model, charmap, MAX_LEN, BATCH_SIZE)

    logP = infer.applyBatch(passwords, TERMINAL_SYMBOL)  # 计算概率
    write_tsv(output_path, passwords, logP)
