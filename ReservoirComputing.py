import copy
import numpy as np

class masking:
    """ This class of function"""
    def __init__(self):
        self.name = masking

    # 生成每个虚拟节点的权重
    def NodeWeight(self,nrow,ncol):
        raw = np.random.rand(nrow,ncol)  # 生成一个nrow行ncol列的、元素满足在0~1之间均匀分布的数组，每一个元素被抽中的概率都是相等的
        new = copy.deepcopy(raw)         # 直接赋值是对象的引用（别名），即浅拷贝，这时候改动某一个别名中的元素都会影响对象本身
                                         # 因此，要实现将数组复制并防止交叉影响，需要深拷贝
        for i in range(nrow):            # 重整化随机数矩阵，使其的值只有1跟-1
            for j in range(ncol):
                if new[i,j] >= 0.5:
                    new[i,j] = 1.0
                else:
                    new[i,j] = -1.0
        return new,raw

class readout:
    """ This class of function serve as the read-out module of Reservoir Computing (RC) algorithm. """
    def __init__(self):
        self.name = readout

    def LinearRegression(self):
        return

if __name__ == '__main__':
    mask = masking()

    a = mask.NodeWeight(10,8)

    print(a[0])
    print(a[1])
