import numpy as np

class evaluate:
    """ This class of function is designed for evaluating the performance of the reservoir computing (RC) model."""
    def __init__(self):
        self.name = evaluate

    # Calculating NRMSE (normalized root-mean-square error)
    # 假设输出为长度为P的向量：y(t) = [y1(t), y2(t), ..., yP(t)]
    # ground_truth的格式[[y1_target(t0), ...yP_target(t0)], ..., [y1_target(tn), ...yP_target(tn)]
    # predicting的格式[[y1_predict(t0), ...yP_predict(t0)], ..., [y1_predict(tn), ...yP_predict(tn)]
    def NRMSE(self,predicting, ground_truth):
        data_length = len(ground_truth)
        predicting = np.array([np.array(predicting[n]) for n in range(data_length)])  # 数据转换
        ground_truth = np.array([np.array(ground_truth[n]) for n in range(data_length)])  # 数据转换
        y_target_average = np.sum(ground_truth, axis=0) / float(data_length)  # 将ground_truth的每一行加起来取平均

        #y = np.sum(ground_truth,axis=0)
        #print(y)

        predicting_deviation = []  # 预测值对于真实值的偏差
        ground_truth_deviation = []  # 真实值本事的离散度
        for i in range(data_length):
            p_value = np.linalg.norm(predicting[i] - ground_truth[i], ord=2)  # 二范数
            predicting_deviation.append(p_value ** 2)  # 二范数的平方

            g_value = np.linalg.norm(ground_truth[i] - y_target_average, ord=2)
            ground_truth_deviation.append(g_value ** 2)

        predicting_deviation = np.array(predicting_deviation)  # 转换成数组
        ground_truth_deviation = np.array(ground_truth_deviation)

        error = np.sqrt(np.sum(predicting_deviation)/np.sum(ground_truth_deviation))  # 求和相除再开根号

        return error