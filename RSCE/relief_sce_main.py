import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utilityScript import getApplicationData, returnImpFeaturesElbow
from FeatureSelectingSCEPyTorch import FeatureSelectingSCE
import pickle
from sklearn.metrics import accuracy_score
from simpleANNClassifierPyTorch import *
# from Relief import relief_feature_selection  # 假设这是 Relief 模块

def relief_feature_selection(data, labels, k, num_samples=50, num_neighbors=10):
    """
    Relief-F特征选择算法
    data: 输入数据 (n_samples, n_features)
    labels: 样本标签 (n_samples,)
    k: 需选择的特征数量
    num_samples: 采样样本数
    num_neighbors: 选取的最近邻个数
    返回: 按权重排序的特征索引列表 (降序)
    """
    n_samples, n_features = data.shape
    num_samples = min(num_samples, n_samples)
    weights = np.zeros(n_features)
    labels = labels.flatten()  # 确保标签为1D

    # 随机采样样本
    sample_indices = np.random.choice(n_samples, size=num_samples, replace=False)

    for i in sample_indices:
        current_sample = data[i]
        current_label = labels[i]

        # 找到同类和异类样本
        same_mask = (labels == current_label)
        same_mask[i] = False  # 排除自身
        same_class = data[same_mask]
        diff_class = data[~same_mask]

        if len(same_class) == 0 or len(diff_class) == 0:
            continue

        # 计算到同类样本的距离并选取 k 个最近邻
        same_dists = np.sqrt(np.sum((same_class - current_sample) ** 2, axis=1))
        same_neighbors = same_class[np.argsort(same_dists)[:num_neighbors]]

        # 计算到异类样本的距离并选取 k 个最近邻
        diff_dists = np.sqrt(np.sum((diff_class - current_sample) ** 2, axis=1))
        diff_neighbors = diff_class[np.argsort(diff_dists)[:num_neighbors]]

        # 更新权重
        for neighbor in same_neighbors:
            weights -= (current_sample - neighbor) ** 2 / num_neighbors
        for neighbor in diff_neighbors:
            weights += (current_sample - neighbor) ** 2 / num_neighbors

    # 归一化并排序
    weights /= num_samples
    selected_indices = np.argsort(weights)[::-1][:k]

    return selected_indices

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

def classifyALLAMLData(partition, featureSet, gpuId):
    trnSet = load_ALLAML_Data('Trn', partition)
    tstSet = load_ALLAML_Data('Tst', partition)
    trData, trLabels = trnSet[:, :-1], trnSet[:, -1]
    tstData, tstLabels = tstSet[:, :-1], tstSet[:, -1]
    trData, tstData = trData[:, featureSet], tstData[:, featureSet]
    print('No. of training samples:', len(trData), 'No. of test samples:', len(tstData))
    print('No. of features used:', len(featureSet))
    
    nClass = len(np.unique(trLabels))
    allACC = []
    allTrainTimes = []  # 存储每次训练时间
    allPredTimes = []   # 存储每次预测时间
    
    for i in range(100):
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
    
    # 计算平均训练和预测时间
    print('all train times:', allTrainTimes)
    print('all pred times:', allPredTimes)
    mean_train_time = np.round(np.mean(allTrainTimes), 4)
    mean_pred_time = np.round(np.mean(allPredTimes), 4)
    
    # 更新打印语句，包含时间信息
    print(f'Data partition: {partition}, Accuracy using all selected features: {mean_accuracy} +/- {std_accuracy}, '
          f'Mean training time: {mean_train_time} s, Mean prediction time: {mean_pred_time} s')
    
    return mean_accuracy

def relief_pretrain(gpuId, trData, trLabels, pp):
    """
    使用 Relief 算法筛选特征并返回选择的特征和准确率。
    
    返回:
        final_features (numpy.ndarray): Relief 选择到的特征索引
        accuracy (float): 分类准确率
    """
    total_features = trData.shape[1]
    print(f"Total number of features: {total_features}")
    k_relief = 200
    relief_indices = relief_feature_selection(trData, trLabels.flatten(), k_relief)
    final_features = relief_indices
    print(f'Number of relief selected features: {len(final_features)}')
    print(f"Selected features by Relief: {final_features}")

    # 使用所有 Relief 选择的特征计算准确率
    accuracy = classifyALLAMLData(pp, final_features, gpuId)
    print("Accuracy with Relief selected features:", accuracy)

    return final_features, accuracy

if __name__ == "__main__":
    # 超参数设置
    num_epochs_pre = 10
    num_epochs_post = 125
    miniBatch_size = 32
    learning_rate = 0.01
    gpuId = 0
    pp = 1  # 数据分区，possible values 0,1,2
    momentum = 0.8
    standardizeFlag = False
    preTrFlag = True

    # 加载训练和测试数据
    trnSet = load_ALLAML_Data('Trn', pp)
    tstSet = load_ALLAML_Data('Tst', pp)
    trData, trLabels = trnSet[:, :-1], trnSet[:, -1]
    tstData, tstLabels = tstSet[:, :-1], tstSet[:, -1]

    # 获取 Relief 选择的特征和准确率
    selected_features, relief_accuracy = relief_pretrain(gpuId, trData, trLabels, pp)

    # 筛选训练和测试数据
    trData = trData[:, selected_features]
    tstData = tstData[:, selected_features]

    # 初始化网络超参数
    dict2 = {}
    dict2['inputDim'] = len(selected_features)  # 输入维度为 200
    dict2['hL'] = [100]
    dict2['hActFunc'] = ['tanh']
    dict2['oActFunc'] = 'linear'
    dict2['errorFunc'] = 'MSE'
    dict2['l1Penalty'] = 0.0002

    # 创建并训练 SCE 模型
    model = FeatureSelectingSCE(dict2)
    model.fit(trData, trLabels, preTraining=preTrFlag, optimizationFunc='Adam', learningRate=learning_rate, m=momentum,
              miniBatchSize=miniBatch_size, numEpochsPreTrn=num_epochs_pre, numEpochsPostTrn=num_epochs_post,
              standardizeFlag=standardizeFlag, verbose=True)

    # 获取 SCE 训练后的特征和准确率
    fWeights = model.fWeight
    fIndices = model.fIndices
    feaList, feaW = returnImpFeaturesElbow(fWeights)
    sce_selected_features = selected_features[fIndices[:len(feaW)]]
    print('Number of selected features by SCE after Relief:', len(feaW))
    print(f"Selected features by SCE after Relief: {sce_selected_features}")

    # 使用所有 SCE 选择的特征计算准确率
    sce_accuracy = classifyALLAMLData(pp, sce_selected_features, gpuId)
    print("Accuracy with Relief + SCE selected features:", sce_accuracy)

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from utilityScript import getApplicationData, returnImpFeaturesElbow
# from FeatureSelectingSCEPyTorch import FeatureSelectingSCE
# import pickle
# from sklearn.metrics import accuracy_score
# from simpleANNClassifierPyTorch import *
# from Relief import relief_feature_selection

# def load_ALLAML_Data(flag='Trn', partition=0):
#     part = 'Partition' + str(partition)
#     if flag.upper() == 'TRN':
#         lData = pickle.load(open('./Data/ALLAML/' + part + '/trnData.p', 'rb'))
#     elif flag.upper() == 'TST':
#         lData = pickle.load(open('./Data/ALLAML/' + part + '/tstData.p', 'rb'))
#     return lData

# import time
# import numpy as np
# from sklearn.metrics import accuracy_score

# def classifyALLAMLData(partition, featureSet, gpuId):
#     trnSet = load_ALLAML_Data('Trn', partition)
#     tstSet = load_ALLAML_Data('Tst', partition)
#     trData, trLabels = trnSet[:, :-1], trnSet[:, -1]
#     tstData, tstLabels = tstSet[:, :-1], tstSet[:, -1]
#     trData, tstData = trData[:, featureSet], tstData[:, featureSet]
    
#     nClass = len(np.unique(trLabels))
#     allACC = []
#     allTrainTimes = []  # 存储每次训练时间
#     allPredTimes = []   # 存储每次预测时间
    
#     for i in range(10):
#         ann = NeuralNet(trData.shape[1], [500], nClass)
        
#         # 测量训练时间
#         start_train_time = time.time()
#         ann.fit(trData, trLabels, standardizeFlag=True, batchSize=200, optimizationFunc='Adam', 
#                 learningRate=0.001, numEpochs=25, cudaDeviceId=gpuId)
#         end_train_time = time.time()
#         train_time = end_train_time - start_train_time
#         allTrainTimes.append(train_time)
        
#         ann = ann.to('cpu')
        
#         # 测量预测时间
#         start_pred_time = time.time()
#         tstPredProb, tstPredLabel = ann.predict(tstData)
#         end_pred_time = time.time()
#         pred_time = end_pred_time - start_pred_time
#         allPredTimes.append(pred_time)
        
#         accuracy = 100 * accuracy_score(tstLabels.flatten(), tstPredLabel)
#         allACC.append(accuracy)
    
#     allACC = np.hstack((allACC))
#     mean_accuracy = np.round(np.mean(allACC), 2)
#     std_accuracy = np.round(np.std(allACC), 2)  # 添加标准差以提供更多信息
    
#     # 计算并打印平均训练和预测时间
#     mean_train_time = np.round(np.mean(allTrainTimes), 4)
#     mean_pred_time = np.round(np.mean(allPredTimes), 4)
    
#     print(f'Data partition: {partition}, Features: {len(featureSet)}, Accuracy: {mean_accuracy} +/- {std_accuracy}, '
#           f'Mean training time: {mean_train_time} s, Mean prediction time: {mean_pred_time} s')
    
#     return mean_accuracy

# def relief_pretrain(gpuId, trData, trLabels, pp, k_relief):
#     total_features = trData.shape[1]
#     relief_indices = relief_feature_selection(trData, trLabels.flatten(), k_relief)
#     final_features = relief_indices
#     return final_features

# if __name__ == "__main__":
#     # 超参数设置
#     num_epochs_pre = 10
#     num_epochs_post = 125
#     miniBatch_size = 32
#     learning_rate = 0.01
#     gpuId = 0
#     pp = 1
#     momentum = 0.8
#     standardizeFlag = False
#     preTrFlag = True

#     # 加载训练和测试数据
#     trnSet = load_ALLAML_Data('Trn', pp)
#     tstSet = load_ALLAML_Data('Tst', pp)
#     trData, trLabels = trnSet[:, :-1], trnSet[:, -1]
#     tstData, tstLabels = tstSet[:, :-1], tstSet[:, -1]

#     # 定义 k_relief 范围，从 100 到 2000，间隔 50
#     k_relief_range = list(range(100, 2001, 20))
#     accuracies = []

#     # 遍历 k_relief
#     for k_relief in k_relief_range:
#         # 获取 Relief 选择的特征
#         selected_features = relief_pretrain(gpuId, trData, trLabels, pp, k_relief)

#         # 筛选训练和测试数据
#         trData_relief = trData[:, selected_features]
#         tstData_relief = tstData[:, selected_features]

#         # 初始化网络超参数
#         dict2 = {}
#         dict2['inputDim'] = len(selected_features)
#         dict2['hL'] = [100]
#         dict2['hActFunc'] = ['tanh']
#         dict2['oActFunc'] = 'linear'
#         dict2['errorFunc'] = 'MSE'
#         dict2['l1Penalty'] = 0.0002

#         # 创建并训练 SCE 模型
#         model = FeatureSelectingSCE(dict2)
#         model.fit(trData_relief, trLabels, preTraining=preTrFlag, optimizationFunc='Adam', learningRate=learning_rate, m=momentum,
#                   miniBatchSize=miniBatch_size, numEpochsPreTrn=num_epochs_pre, numEpochsPostTrn=num_epochs_post,
#                   standardizeFlag=standardizeFlag, verbose=False)

#         # 获取 SCE 训练后的特征和准确率
#         fWeights = model.fWeight
#         fIndices = model.fIndices
#         feaList, feaW = returnImpFeaturesElbow(fWeights)
#         sce_selected_features = selected_features[fIndices[:len(feaW)]]

#         # 计算 Relief + SCE 的准确率
#         sce_accuracy = classifyALLAMLData(pp, sce_selected_features, gpuId)
#         print(f"Accuracy with Relief + SCE selected features for k_relief={k_relief}: {sce_accuracy}")
#         accuracies.append(sce_accuracy)

#     # 绘制折线图
#     plt.figure(figsize=(10, 6))
#     plt.plot(k_relief_range, accuracies, marker='o', linestyle='-', color='b')
#     plt.xlabel('Number of Features Selected by Relief')
#     plt.ylabel('Accuracy (%) after SCE')
#     plt.title('Accuracy of Relief-SCE')
#     plt.grid(True)
#     plt.xticks(np.arange(100, 2100, 200))  # 设置横轴刻度
#     plt.tight_layout()
#     plt.savefig('relief_sce_accuracy.png')  # 保存图像
#     plt.show()
#     plt.close()