# 保存与导入pickle文件
import pickle


def dump(filename, data, **kargs):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load(filename, python2=False, **kargs):
    """
    加载pickle文件
    Args:
        filename: 导入文件名
        python2: 版本是否为python2
        **kargs: 其他参数
    Returns: pickle文件内容
    """
    if python2:
        kargs.update(encoding='latin1')
    with open(filename, 'rb') as f:
        data = pickle.load(f, **kargs)
    return data
