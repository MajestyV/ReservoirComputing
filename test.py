import numpy as np
import matplotlib.pyplot as plt
from ReservoirComputingTesting import ReservoirComputing
from ReservoirComputingTesting import DynamicSystems
from ReservoirComputingTesting import Evaluation

RC_masking = ReservoirComputing.masking()
RC_reservoir = ReservoirComputing.reservoir()
RC_readout = ReservoirComputing.readout()

DS = DynamicSystems.dynamic_systems()

ev = Evaluation.evaluate()

#data = np.array([[ -2.95507616,  10.94533252],
                 #[ -0.44226119,   2.96705822],
                 #[ -2.13294087,   6.57336839],
                 #[  1.84990823,   5.44244467],
                 #[  0.35139795,   2.83533936],
                 #[ -1.77443098,   5.6800407 ],
                 #[ -1.8657203 ,   6.34470814],
                 #[  1.61526823,   4.77833358],
                 #[ -2.38043687,   8.51887713],
                 #[ -1.40513866,   4.18262786]])

#a = np.zeros([2,6])
    #a[0] = np.array([1,1,1,1,1,1])
    #print(a)

#a = ds.LorenzSystem([3.051522,1.582542,15.62388],[10.0,29,2.667],100000,0.0005)
#a = ds.ChuaCircuit([0.1,0.1,0.1],[10,12.33,-0.544,-1.088],500000,0.001)  # 双漩涡混沌
#a = ds.ChuaCircuit([0.1, 0.1, 0.1], [10, 19.7226, -0.688, -1.376], 10000, 0.01)  # 混沌-周期-混沌-周期演变

N = 4
Q = 3

ground_truth_raw = DS.ChuaCircuit([0.1, 0.1, 0.1], [10, 19.7226, -0.688, -1.376], 20000, 0.01)  # 混沌-周期-混沌-周期演变
ground_truth = DS.Rearrange(ground_truth_raw)  # 数据重整化

# 前3000点可能包含初始点的信息，会是我们的拟合偏移，因此我们从3000点之后开始取值
training_input = ground_truth_raw[1000:6000]
training_output = ground_truth_raw[1001:6001]

# 预测
predicting_input = ground_truth_raw[6001:15000]
predicting_output = ground_truth_raw[6002:15001]

training_input_new = RC_masking.Normalizing(training_input,(0,5),shift_vec_manual="True",shift_vec=np.array([3.0,3.0,3.0]))

predicting_input_new = RC_masking.Normalizing(predicting_input,(0,5),shift_vec_manual="True",shift_vec=np.array([3.0,3.0,3.0]))

M = RC_masking.NodeWeight(Q,N)

M_fixed_regulated = np.array([[-1.,  1., -1., -1.],
                              [-1., -1.,  1.,  1.],
                              [-1.,  1.,  1.,  1.]])

M_fixed = np.array([[0.98345375, 0.61861715, 0.56612098, 0.39718429],
                    [0.5376763,  0.34286099, 0.26059168, 0.36128826],
                    [0.0365375,  0.42064209, 0.07613754, 0.5211719 ]])

#M_fixed_1 =

print(M)
J = np.dot(training_input_new,M)

J_predict = np.dot(predicting_input_new,M)

# X = RC_reservoir.Sigmoid(J)
X = RC_reservoir.I_nonlinear(J,0,1,1,1)

X_predict = RC_reservoir.I_nonlinear(J_predict,0,1,1,1)

#print(X)

coef, intercept = RC_readout.LASSO(X,training_output)

testing_output = np.dot(X,np.transpose(coef))+intercept

predicted_output = np.dot(X_predict,np.transpose(coef))+intercept
#print(testing_output)
#print(training_output)

testing_output_rearrange = DS.Rearrange(testing_output)
t_testing = testing_output_rearrange[0]
x_testing = testing_output_rearrange[1]
y_testing = testing_output_rearrange[2]
z_testing = testing_output_rearrange[3]

training_output_rearrange = DS.Rearrange(training_output)
t_traning = training_output_rearrange[0]
x_traning = training_output_rearrange[1]
y_traning = training_output_rearrange[2]
z_traning = training_output_rearrange[3]

predicted_output_rearrange = DS.Rearrange(predicted_output)
t_predicted = predicted_output_rearrange[0]
x_predicted = predicted_output_rearrange[1]
y_predicted = predicted_output_rearrange[2]
z_predicted = predicted_output_rearrange[3]

predicting_output_rearrange = DS.Rearrange(predicting_output)  # 标准数据-ground truth
t_predicting = predicting_output_rearrange[0]
x_predicting = predicting_output_rearrange[1]
y_predicting = predicting_output_rearrange[2]
z_predicting = predicting_output_rearrange[3]

#plt.plot(x_testing,z_testing)
#plt.plot(x_traning,z_traning)

#print(len(predicted_output))
#print(len(predicting_output))

nrmse = ev.NRMSE(predicted_output,predicting_output)
print(nrmse)

plt.plot(x_predicted,z_predicted)
plt.plot(x_predicting,z_predicting)

#print(coef)
#print(intercept)


#print(training_input_new)
#print(M)
#print(J)
#print(training_output)

#print(ground_truth)
#print(ground_truth_raw)

#print(len(ground_truth_raw[3000:6000]))

#ax = plt.subplot(projection='3d')
#ax.plot(x,y,z)
    #plt.xlim(-1,1)

    #b = ds.Rearrange(a)

    #plt.plot(b[0],b[1])

    #print(np.linspace(0,10-1,10))

# 在sklearn中，所有的数据都应该是二维矩阵，哪怕它只是单独一行或一列，所以需要使用numpy库的reshape(1,-1)进行转换
# reshape(1,-1)转化成1行
# reshape(2,-1)转换成2行
# reshape(-1,1)转换成1列
# reshape(-1,2)转化成2列
# reshape(2,8)转化成2行8列
#X = data[:,0].reshape(-1,1)
#Y = data[:,1].reshape(-1,1)

#print(X,Y)

#print(RC_readout.LASSO(X,Y))

#a = np.linspace(-5,5,100)
#plt.plot(a,-0.91831979*a+4.98752105)
#plt.scatter(data[:,0],data[:,1])

