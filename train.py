import os
import sys
import gin
# gin可提供/修改函数参数默认值，通过@gin.configurable将函数参数和.gin文件绑定
# 文件语法：function_name.parameter_name = value
# 除非指定参数，否则调用函数时参数由.gin文件提供
# gin也可提供class的__init__参数，通过 @gin.configurable  class
# 通过：gin.parse_config_file('config.gin')来连接.gin文件内容
import tensorflow as tf
import modelCNN as model
import input_pipeline
from trainer import Trainer

gpus = tf.config.experimental.list_physical_devices('GPU')

MODEL_OUT = 'HOME/MODELs'
LOG_OUT = 'HOME/LOGs'


def basenameNoExt(path, sep='.'):
    name = os.path.basename(path)  # 返回 path 的最后一级目录，即文件名
    return name.split(sep)[0]  # 去除文件名的后缀


@gin.configurable
def setup(name, MODEL_TYPE, home_train, home_tests, max_epochs, log_freq, MAX_LEN, hparams):
    # 根据.gin初始化参数
    check_train_dir = os.path.join(LOG_OUT, name)
    check_test_dir = os.path.join(check_train_dir, 'eval')

    MAX_MASK = hparams['masked_chars']
    INCLUDE_END_SYMBOL = hparams['append_end']
    # 数据集获取
    train_batch, vocab_size, charmap = input_pipeline.makeIterInput(home_train, hparams['batch_size'], MAX_MASK,
                                                               INCLUDE_END_SYMBOL, MAX_LEN)
    # 设置optimizer
    optimizer = tf.keras.optimizers.Adam(hparams['learning_rate'])
    # 模型结构
    pretrain_model, train_step, predict_step = model.make_train_predict(hparams, optimizer, vocab_size, MAX_LEN)  # f:tf.keras.Model

    model_mem_footprint = (pretrain_model.count_params() * 4) // (10 ** 6)
    print("\t MODEL_MEM: %dMB" % model_mem_footprint)
    # prefetch预取数据,将生成数据的时间和使用数据的时间分离,在请求元素之前从输入数据集中预取这些元素
    t = Trainer(
        pretrain_model,
        MAX_LEN,
        charmap,
        train_step,
        predict_step,
        max_epochs,
        train_batch,
        home_tests,
        optimizer,
        check_train_dir,
        check_test_dir,
        1,
        log_freq,
        hparams,
    )
    # 训练
    print("TRAIN")
    t()  # Trainer类中有__call__，则可用此方法调用
    print("EXPORT")
    mpath = os.path.join(MODEL_OUT, name + '.h5')
    pretrain_model.save(mpath, overwrite=True, include_optimizer=False, save_format='h5')  # 以h5格式保存模型参数（其他格式还有tf的SavedModel）


if __name__ == '__main__':
    try:
        conf_path = sys.argv[1]  # sys.argv是一个字符串的列表，包含了命令行参数
    except:
        print("USAGE: conf_file_gin")
        sys.exit(1)

    gin.parse_config_file(conf_path)  # 参见前面的注释

    name = basenameNoExt(conf_path)
    print("Name: ", name)

    setup(name)
