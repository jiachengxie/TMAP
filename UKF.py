import copy
import json
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
import torch
def transition_function(x, dt):
    # 状态转移矩阵
    F = np.array([[1, 0, dt, 0, 0],
                  [0, 1, 0, dt, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    return np.dot(F, x)


def measurement_function(x):
    # 测量函数
    H = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0]])
    return np.dot(H, x)
def create_ukf(dim_x, dim_z, dt):
    # 定义无迹卡尔曼滤波的sigma点
    points = MerweScaledSigmaPoints(n=dim_x, alpha=0.1, beta=2.0, kappa=1.0)

    # 创建无迹卡尔曼滤波器
    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, points=points,hx=measurement_function,fx=transition_function)

    return ukf

def save_ukf_params(ukf, filename):
    # 将滤波器参数保存为json文件
    params = {
        'dim_x': ukf._dim_x,
        'dim_z': ukf._dim_z,
        'dt': ukf._dt,
        'Q': ukf.Q.tolist(),
        'R': ukf.R.tolist(),
        'x': ukf.x.tolist(),
        'P': ukf.P.tolist()
    }
    with open(filename, 'w') as file:
        json.dump(params, file)

def load_ukf_params(filename):
    # 从json文件中加载滤波器参数并创建滤波器
    with open(filename, 'r') as file:
        params = json.load(file)
    ukf = create_ukf(params['dim_x'], params['dim_z'], params['dt'])
    ukf.Q = np.array(params['Q'])
    ukf.R = np.array(params['R'])
    ukf.x = np.array(params['x'])
    ukf.P = np.array(params['P'])
    return ukf

def predict_next_state(ukf, traj_len):
    # 预测下一个轨迹长度的状态点
    predicted_states = []
    for _ in range(traj_len):
        ukf.predict()
        predicted_states.append(ukf.x.copy())
    return np.array(predicted_states)

def correct_nn_prediction(ukf, nn_prediction,traj_len,a): #nn_prediction:(num_samples, len_traj/3*2)
    # 用预测的状态点修正神经网络的当前预测
    filtered_x = np.copy(nn_prediction)
    ukf_pre = predict_next_state(ukf,traj_len)
    ukf_p = ukf_pre[0::3,:] # (len_traj/3,2)
    ukf_p = ukf_p.reshape(1,-1) # (1,len_traj/3*2)
    ukf_p1 = np.repeat(np.copy(ukf_p),len(nn_prediction),axis=0)
    filtered_x = a * ukf_p1 + (1-a)*filtered_x
    return filtered_x,ukf_p

def update_ukf(ukf, true_measurements):
    # 根据当前真实值更新滤波器
    for measurement in true_measurements:
        for mea in measurement:
            ukf.update(mea)
    return ukf

def correct(ukf,x,Z):#Z是神经网络的预测结果
    H = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0]])
    P = ukf.P
    R = ukf.R
    K = copy.deepcopy(P) @ np.transpose(copy.deepcopy(H)) @ np.linalg.inv(copy.deepcopy(H) @ copy.deepcopy(P) @ np.transpose(copy.deepcopy(H))+copy.deepcopy(R))
    X = np.copy(x) + np.copy(K) @ (np.copy(Z) - np.copy(x[0:2]))
    return X[:2]