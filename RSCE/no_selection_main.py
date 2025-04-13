import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utilityScript import getApplicationData, returnImpFeaturesElbow
from FeatureSelectingSCEPyTorch import FeatureSelectingSCE, createOutputAsCentroids  # 引入 centroid 计算函数
import pickle
from sklearn.metrics import accuracy_score
from simpleANNClassifierPyTorch import *

def load_ALLAML_Data(flag = 'Trn', partition=0):
    # load data file
    part = 'Partition' + str(partition)
    if flag.upper() == 'TRN':
        lData = pickle.load(open('G:/Machine Learning/ECE271A_Project_GPU/ECE271_project-main/Data/ALLAML/' + part + '/trnData.p', 'rb'))
    elif flag.upper() == 'TST':
        lData = pickle.load(open('G:/Machine Learning/ECE271A_Project_GPU/ECE271_project-main/Data/ALLAML/'+ part + '/tstData.p', 'rb'))
    
    return lData

import time
import numpy as np
from sklearn.metrics import accuracy_score

def classifyALLAMLData(partition, trData, tstData, gpuId):
    # 不进行特征选择，直接使用所有特征
    trLabels = trData[:, -1]
    tstLabels = tstData[:, -1]
    trData, tstData = trData[:, :-1], tstData[:, :-1]
    
    nClass = len(np.unique(trLabels))
    allACC = []
    allTrainTimes = []  # 存储每次训练时间
    allPredTimes = []   # 存储每次预测时间
    
    for i in range(10):
        ann = NeuralNet(trData.shape[1], [500], nClass)
        
        # 测量训练时间
        start_train_time = time.time()
        ann.fit(trData, trLabels, standardizeFlag=True, batchSize=200, optimizationFunc='Adam', 
                learningRate=0.001, numEpochs=25, cudaDeviceId=gpuId)
        end_train_time = time.time()
        train_time = end_train_time - start_train_time
        allTrainTimes.append(train_time)
        
        ann = ann.to('cpu')
        
        # 测量预测时间
        start_pred_time = time.time()
        tstPredProb, tstPredLabel = ann.predict(tstData)
        end_pred_time = time.time()
        pred_time = end_pred_time - start_pred_time
        allPredTimes.append(pred_time)
        
        accuracy = 100 * accuracy_score(tstLabels.flatten(), tstPredLabel)
        allACC.append(accuracy)
    
    allACC = np.hstack((allACC))
    mean_accuracy = np.round(np.mean(allACC), 2)
    std_accuracy = np.round(np.std(allACC), 2)
    
    # 计算并打印平均训练和预测时间
    mean_train_time = np.round(np.mean(allTrainTimes), 4)
    mean_pred_time = np.round(np.mean(allPredTimes), 4)
    
    print(f'Data partition: {partition}, Accuracy: {mean_accuracy} +/- {std_accuracy}, '
          f'Mean training time: {mean_train_time} s, Mean prediction time: {mean_pred_time} s')
    
    return mean_accuracy, std_accuracy  # 只返回准确率和标准差

def calculate_mse_to_centroids(trData, trLabels):
    """
    计算所有特征到类中心（centroid）的均方误差（MSE）距离。
    """
    centroids = createOutputAsCentroids(trData, trLabels)
    mse_total = 0.0
    n_samples = trData.shape[0]
    
    for i in range(n_samples):
        sample = trData[i]
        centroid = centroids[i]
        mse_total += np.mean((sample - centroid) ** 2)
    
    mse = mse_total / n_samples
    return mse

if __name__ == "__main__":
    # 超参数设置
    gpuId = 0
    pp = 1  # 数据分区，possible values 0,1,2

    # 加载训练和测试数据
    trnSet = load_ALLAML_Data('Trn', pp)
    tstSet = load_ALLAML_Data('Tst', pp)
    trData, trLabels = trnSet[:, :-1], trnSet[:, -1]
    tstData, tstLabels = tstSet[:, :-1], tstSet[:, -1]

    # 计算所有特征到质心的 MSE
    mse = calculate_mse_to_centroids(trData, trLabels)
    
    # 不进行特征选择，直接使用所有特征计算分类准确率
    mean_accuracy, std_accuracy = classifyALLAMLData(pp, trnSet, tstSet, gpuId)

    # 输出结果
    print(f"Total number of features: {trData.shape[1]}")
    print(f"MSE distance to centroids (all features): {mse:.6f}")
    print(f"Accuracy using all features: {mean_accuracy} +/- {std_accuracy}")