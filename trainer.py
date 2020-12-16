import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
import tqdm, os
from glob import glob
from inference import Inference
from evaluate import getScore, readPC, rank

patience = 5
PROFILE_ITER = 1000
BATCH_EVAL = 2046


def flush_metric(iteration, metric, non_scalar=False):
    name = metric.name
    if non_scalar:
        value = metric.result()[0]
    else:
        value = metric.result()
    tf.summary.scalar(name, value, step=iteration)
    metric.reset_states()
    return (name, value)


flatten = lambda x: tf.reshape(x, (-1, 1))


class Trainer:
    def __init__(self,
                 AE_model,
                 max_len,
                 charmap,
                 train_step,
                 predict_step,
                 epochs,
                 train_batch,
                 home_tests,
                 optimizer,
                 log_train,
                 log_test,
                 test_num_steps,
                 log_freq,
                 hparams
                 ):

        self.AE_model = AE_model
        self.max_len = max_len
        self.charmap = charmap
        self.home_tests = home_tests
        self.train_step = train_step
        self.predict_step = predict_step
        self.epochs = epochs

        self.train_batch = train_batch
        self.optimizer = optimizer
        self.log_freq = log_freq

        self.hparams = hparams

        self.test_num_steps = test_num_steps

        # early stopping
        self.top_score = None
        self.countdown = patience

        self.log_train = log_train

        # Creates a summary file writer for the given log directory.
        self.train_summary_writer = train_summary_writer = tf.summary.create_file_writer(log_train)
        if log_test:
            self.test_summary_writer = test_summary_writer = tf.summary.create_file_writer(log_test)

        # check points
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.AE_model,
        )  # 可追踪变量以二进制的方式储存成一个.ckpt文件;键值对形式窜入参数，键名随意

        # 可以设置自动存点间隔步数、最大断点数、自动存点间隔时间等
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint, log_train, max_to_keep=patience)

        if self.ckpt_manager.latest_checkpoint:  # 加载最新的断点
            checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def train_and_test(self, dataset, profile=False):
        """
        训练
        Args:
            dataset: 来自input_pipeline.py的数据管道
            profile:

        Returns:
        """
        # create metrics
        loss_m = tf.keras.metrics.Mean(name='loss')  # 用法的流程：1、新建一个metric  2、更新数据update_state  3、取出数据result().numpy()  4、 重置清零数据reset_states
        acc_m = tf.keras.metrics.Accuracy(name='accuracy')

        for i in range(self.epochs):
            print(f'Epoch {i}')
            for data in tqdm.tqdm(dataset):  # Tqdm 是 Python 进度条库，可以在 Python 长循环中添加一个进度提示信息，只需在.tqdm中要封装任意的迭代器

                x, prediction_mask, y = data

                iteration = self.optimizer.iterations  # 迭代次数

                loss, p, prediction = self.train_step(data)  # 训练一批data

                loss_m.update_state(loss)
                # acc_m.update_state(_y, _prediction)

                if tf.equal(iteration % self.log_freq, 0):  # 用于tensorboard的显示
                    flush_metric(iteration, loss_m)
                    # flush_metric(iteration, acc_m)

            if profile:
                print("END PROFILE")
                break

            with self.test_summary_writer.as_default():
                # application task test
                rc_scores = self.rankConfOnTestSets(BATCH_EVAL)
                avg_score = 0
                for name, score in rc_scores:
                    print(name, score)
                    avg_score += score[0]
                    tf.summary.scalar(f'WRankCof_{name}', score[0], step=i + 1)
                avg_score = avg_score / len(rc_scores)

                tf.summary.scalar(f'WRankCof_avg', avg_score, step=i + 1)

                # early_stop
                if self.early_stopping(-avg_score):
                    print(f"Early-stop epoch-{i}")
                    break

    def rankConfOnTestSets(self, batch_size):
        inf = Inference(self.AE_model, self.charmap, self.max_len, batch_size)

        scores = []
        paths = glob(self.home_tests)
        for path in paths:
            name = os.path.basename(path).split('-')[0]
            print(name)
            X, F = readPC(path, self.max_len - 1, encoding='ascii')
            R = rank(F)
            R = np.array(R)

            # apply model
            UP = inf.applyBatch(X, INCLUDE_END_SYMBOL=self.hparams['append_end'])
            UP = np.array(UP)
            score = getScore(UP, R)
            scores += [(name, score)]

        return scores

    def early_stopping(self, test_score):
        print(test_score)
        if self.top_score is None or test_score < self.top_score:
            print("New Best", test_score)
            self.top_score = test_score
            self.countdown = patience

            # Save checkpoint
            ckpt_save_path = self.ckpt_manager.save()
            print('Saving checkpoint at {}'.format(ckpt_save_path))

            return False

        self.countdown -= 1
        if self.countdown == 0:
            print("UTB", test_score, self.top_score)
            return True

    def __call__(self, profile_run=False):
        if profile_run:
            with tf.profiler.experimental.Profile(self.log_train):
                self.train(self.train_batch.take(PROFILE_ITER), True)
        else:
            # with启动的上下文管理器内部所定义的tf.Operation则会添加进入当前的default_writer中
            with self.train_summary_writer.as_default():
                self.train_and_test(self.train_batch)
                hp.hparams(self.hparams)
