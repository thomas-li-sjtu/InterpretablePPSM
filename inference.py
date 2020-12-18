import tensorflow as tf
import numpy as np
from input_pipeline import MASK, string2idx, idx2string
import tqdm


class Inference:
    def __init__(self, model, charmap, max_len, batch_size):
        self.model = model
        self.max_len = max_len
        self.batch_size = batch_size

        self.charmap = charmap
        self.cm_ = np.array([x[0] for x in sorted(self.charmap.items(), key=lambda x: x[1])])

    def __call__(self, s, INCLUDE_END_SYMBOL):

        oxi, xi = self.makeHoles(s, INCLUDE_END_SYMBOL)
        i = np.arange(len(xi[0]))[None, :]

        out = self.model([xi, i], training=False)
        p = out[1]

        un_p = self.un_p(oxi, p)

        return un_p

    def makeHoles(self, x, INCLUDE_END_SYMBOL):
        # 将口令x替换为数字串 oxi，并轮流将每一位替换为mask，得到二维数组xi
        xl = len(x)
        oxi = string2idx(x, self.charmap, self.max_len, 0, INCLUDE_END_SYMBOL)
        xi = np.tile(oxi, (xl, 1))
        for i in range(xl):
            xi[i, i] = MASK
        return oxi, xi

    def un_p(self, xi, p, scalar=False):
        mp = np.zeros(len(p))  # p的长度由当前的口令xi长度决定
        for i in range(len(p)):
            t = xi[i]
            mp[i] = p[i][i][t]

        if scalar:
            return np.log(mp).sum()

        return mp

    def _alppyBatch(self, H, passwords_strtoindx, length_passwords):

        n = len(passwords_strtoindx)

        I = np.arange(len(H[0]))[None, :]
        I = np.tile(I, (len(H), 1))
        out = self.model([H, I], training=False)  # 输入mask后的数组
        logits_ = out[1]  # softmax的输出

        SCOREs = [None] * n
        tot = 0

        for i in range(n):
            length_password = length_passwords[i]
            logit_i = logits_[tot:tot + length_password]  # 某一条口令的softmax结果：二维张量
            assert len(logit_i) == length_password  #
            SCOREs[i] = self.un_p(passwords_strtoindx[i], logit_i, scalar=True)
            tot += length_password
        return SCOREs

    def applyBatch(self, passwords, INCLUDE_END_SYMBOL):
        H = []  # make hole之后
        passwords_strtoindx = []  # 口令映射为数字串
        length_passwords = []  # 口令长度
        SCORE = []

        for index, tmp_password in tqdm.tqdm(list(enumerate(passwords))):
            oxi, xi = self.makeHoles(tmp_password, INCLUDE_END_SYMBOL)
            length_passwords.append(len(tmp_password))
            passwords_strtoindx.append(oxi)
            H.append(xi)

            if index and index % self.batch_size == 0:  # 达到一个batch_size的长度
                H = np.concatenate(H)
                SCORE += self._alppyBatch(H, passwords_strtoindx, length_passwords)
                H = []
                passwords_strtoindx = []
                length_passwords = []
        if H:
            H = np.concatenate(H)
            SCORE += self._alppyBatch(H, passwords_strtoindx, length_passwords)

        return SCORE

    def sorting(self, X):
        up = np.array(self.applyBatch(X))
        perm = np.argsort(-up)
        return perm
