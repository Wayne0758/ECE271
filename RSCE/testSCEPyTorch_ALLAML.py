import pdb
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utilityScript import getApplicationData, returnImpFeaturesElbow
from FeatureSelectingSCEPyTorch import FeatureSelectingSCE
import pickle
from sklearn.metrics import accuracy_score
from simpleANNClassifierPyTorch import *
from Relief import relief_feature_selection

def load_ALLAML_Data(flag = 'Trn', partition=0):
    # load data file
    part = 'Partition' + str(partition)
    if flag.upper() == 'TRN':
        lData = pickle.load(open('G:/Machine Learning/ECE271A_Project_GPU/ECE271_project-main/Data/ALLAML/' + part + '/trnData.p', 'rb'))
    elif flag.upper() == 'TST':
        lData = pickle.load(open('G:/Machine Learning/ECE271A_Project_GPU/ECE271_project-main/Data/ALLAML/'+ part + '/tstData.p', 'rb'))
    
    return lData

def classifyALLAMLData(partition, featureSet, fCntList, gpuId):
    accuracyList = []
    display = True
    for feaCnt in fCntList:
        # load ALLAML data
        trnSet = load_ALLAML_Data('Trn', partition)
        tstSet = load_ALLAML_Data('Tst', partition)
        trData, trLabels = trnSet[:, :-1], trnSet[:, -1]
        tstData, tstLabels = tstSet[:, :-1], tstSet[:, -1]
        fea = featureSet[:feaCnt]

        # use the selected features
        trData, tstData = trData[:, fea], tstData[:, fea]
        if display:
            print('No. of training samples', len(trData), ' No of test samples', len(tstData))
            display = False

        nClass = len(np.unique(trLabels))
        allACC = []
        for i in range(10):
            ann = NeuralNet(trData.shape[1], [500], nClass)
            ann.fit(trData, trLabels, standardizeFlag=True, batchSize=200, optimizationFunc='Adam', learningRate=0.001,
                    numEpochs=25, cudaDeviceId=gpuId)
            ann = ann.to('cpu')
            tstPredProb, tstPredLabel = ann.predict(tstData)
            accuracy = 100 * accuracy_score(tstLabels.flatten(), tstPredLabel)
            allACC.append(accuracy)
            # print('Accuracy after iteration', i+1, np.round(accuracy, 2))
        allACC = np.hstack((allACC))
        # print('No. of features:', trData.shape[1], '.Average Accuracy:', np.round(np.mean(allACC), 2), '+/', np.round(np.std(allACC), 2))
        accuracyList.append(np.round(np.mean(allACC), 2))
        # pdb.set_trace()
        print('Data partition:', partition, 'Accuracy using', feaCnt, 'no. of features:', np.round(np.mean(allACC), 2),
              '+/', np.round(np.std(allACC), 2))
    return accuracyList

def classifyWithRelief(partition, featureSet, fCntList, gpuId):
    """
    新分类函数：使用Relief筛选后的特征
    不修改原有 classifyALLAMLData，保持其逻辑不变
    """
    accuracyList = []
    display = True
    
    for feaCnt in fCntList:
        # 加载数据（与原有函数一致）
        trnSet = load_ALLAML_Data('Trn', partition)
        tstSet = load_ALLAML_Data('Tst', partition)
        trData, trLabels = trnSet[:, :-1], trnSet[:, -1]
        tstData, tstLabels = tstSet[:, :-1], tstSet[:, -1]
        fea = featureSet[:feaCnt]  # 直接使用传入的最终特征列表
        
        if display:
            print('No. of training samples', len(trData), 'No. of test samples', len(tstData))
            display = False
        
        trData, tstData = trData[:, fea], tstData[:, fea]
        nClass = len(np.unique(trLabels))
        allACC = []
        allTrainTimes = []  # 存储每次训练时间
        allPredTimes = []   # 存储每次预测时间
        
        for i in range(100):
            ann = NeuralNet(trData.shape[1], [500], nClass)
            
            # 测量训练时间
            start_train_time = time.time()
            ann.fit(trData, trLabels, standardizeFlag=True, batchSize=200, 
                    optimizationFunc='Adam', learningRate=0.001, numEpochs=25, 
                    cudaDeviceId=gpuId)
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
        mean_acc = np.round(np.mean(allACC), 2)
        std_acc = np.round(np.std(allACC), 2)
        print('all train times:', allTrainTimes)
        print('all pred times:', allPredTimes)
        mean_train_time = np.round(np.mean(allTrainTimes), 4)
        mean_pred_time = np.round(np.mean(allPredTimes), 4)
        
        accuracyList.append(mean_acc)
        
        print(f'Data partition: {partition}, Features: {feaCnt}, Accuracy: {mean_acc} +/- {std_acc}, '
              f'Mean training time: {mean_train_time} s, Mean prediction time: {mean_pred_time} s')
    
    return accuracyList  # 只返回准确率列表

if __name__ == "__main__":
    # hyper-parameters for Adam optimizer
    num_epochs_pre = 10
    num_epochs_post = 125
    # for high dimensional dataset use a single minibatch. I did this by setting the minibatch size large.
    miniBatch_size = 32
    learning_rate = 0.01
    gpuId = 0
    pp = 1  # possible values 0,1,2
    momentum = 0.8
    standardizeFlag = False
    preTrFlag = True

    # load training data data
    # label 0 is ALL and label 1 is AML
    trnSet = load_ALLAML_Data('Trn', pp)
    tstSet = load_ALLAML_Data('Tst', pp)
    trData, trLabels = trnSet[:, :-1], trnSet[:, -1]
    tstData, tstLabels = tstSet[:, :-1], tstSet[:, -1]

    # initialize network hyper-parameters
    dict2 = {}
    dict2['inputDim'] = np.shape(trData)[1]
    dict2['hL'] = [100]
    dict2['hActFunc'] = ['tanh']
    dict2['oActFunc'] = 'linear'
    dict2['errorFunc'] = 'MSE'
    dict2['l1Penalty'] = 0.0002
    model = FeatureSelectingSCE(dict2)
    model.fit(trData, trLabels, preTraining=preTrFlag, optimizationFunc='Adam', learningRate=learning_rate, m=momentum,
              miniBatchSize=miniBatch_size,
              numEpochsPreTrn=num_epochs_pre, numEpochsPostTrn=num_epochs_post, standardizeFlag=standardizeFlag,
              verbose=True)

    # fWeights = model.fWeight.to('cpu').numpy()
    # fIndices = model.fIndices.to('cpu').numpy()
    fWeights = model.fWeight
    fIndices = model.fIndices
    feaList, feaW = returnImpFeaturesElbow(fWeights)
    print('No of selected features', len(feaW))
    fCntList = [10, 50]
    # using the selected features run classification
    accuracyList = classifyALLAMLData(pp, feaList, fCntList, gpuId)


    # k_relief = 60
    # trData_screened = trData[:, feaList]
    # relief_indices = relief_feature_selection(trData_screened, trLabels.flatten(), k_relief)

    # final_features = feaList[relief_indices]
    # print('Number of relief selected features:', len(final_features))

    # fCntList = [k_relief]
    # accuracyListRelief = classifyWithRelief(pp, final_features, fCntList, gpuId)

    # print("\n对比结果：")
    # print("- SCE选特征后准确率:", accuracyList)
    # print("- SCE+Relief后准确率:", accuracyListRelief)



    # 假设 feaW 已定义，例如 feaW 是特征权重列表或特征总数
    # 如果 feaW 未定义，可以改为 len(feaList)
    max_k_relief = len(feaW)  # 最大 k_relief 值

    # 定义 k_relief 的范围，从 10 到 len(feaW)
    k_relief_range = range(10, max_k_relief + 1)  # +1 是因为 range 是左闭右开

    # 用于保存结果的列表
    results = []
    k_relief_values = []  # 用于存储 k_relief 值
    accuracy_values = []  # 用于存储准确率值

    # 遍历所有 k_relief 值
    trData_screened = trData[:, feaList]  # 这部分提到循环外，避免重复计算
    for k_relief in k_relief_range:
        try:
            # 调用 Relief 特征选择
            relief_indices = relief_feature_selection(trData_screened, trLabels.flatten(), k_relief)
            final_features = feaList[relief_indices]
            print(f'Number of relief selected features for k_relief={k_relief}: {len(final_features)}')

            # 分类并获取准确率
            fCntList = [k_relief]
            accuracyListRelief = classifyWithRelief(pp, final_features, fCntList, gpuId)
            accuracy = accuracyListRelief[0]  # 假设返回的是列表，取第一个值

            # 保存结果
            results.append((k_relief, accuracy))
            k_relief_values.append(k_relief)
            accuracy_values.append(accuracy)
            print(f"k_relief={k_relief}, Accuracy={accuracy}")

        except ValueError as e:
            print(f"Error at k_relief={k_relief}: {e}")
            break  # 如果出错（比如样本数不够），停止循环

    # 将结果保存到文本文件
    output_file = "relief_accuracy_results.txt"
    with open(output_file, 'w') as f:
        for k_relief, accuracy in results:
            f.write(f"k_relief: {k_relief}, accuracy: {accuracy}\n")

    # 可视化：绘制折线图并添加两条水平线
    plt.figure(figsize=(12, 6))  # 设置图形大小
    plt.plot(k_relief_values, accuracy_values, marker='o', linestyle='-', color='b', label='Relief Accuracy')

    # 添加 accuracyList 的两条水平线
    # 假设 accuracyList 已定义，包含两个值：[top-10 准确率, top-50 准确率]
    plt.axhline(y=accuracyList[0], color='r', linestyle='--', label='Top-10 Features Accuracy')
    plt.axhline(y=accuracyList[1], color='g', linestyle='--', label='Top-50 Features Accuracy')

    # 设置标签和标题
    plt.xlabel('k_relief (Number of Features)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs k_relief with Relief Feature Selection')
    plt.grid(True)  # 添加网格线
    plt.legend()  # 添加图例
    plt.tight_layout()

    # 保存折线图到文件
    plt.savefig('relief_accuracy_plot_with_baselines.png')
    plt.show()  # 显示图形

    # 打印对比结果（可选）
    print("\n最终结果已保存到", output_file)
    print("- SCE选特征后准确率:", accuracyList)
    print("- SCE+Relief后准确率 (最后一个 k_relief):", accuracyListRelief)