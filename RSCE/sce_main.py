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


def load_ALLAML_Data(flag='Trn', partition=0):
    # load data file
    part = 'Partition' + str(partition)
    if flag.upper() == 'TRN':
        lData = pickle.load(open('./Data/ALLAML/' + part + '/trnData.p', 'rb'))
    elif flag.upper() == 'TST':
        lData = pickle.load(open('./Data/ALLAML/' + part + '/tstData.p', 'rb'))

    return lData


import time
import numpy as np
from sklearn.metrics import accuracy_score

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
            # print('Accuracy after iteration', i+1, np.round(accuracy, 2))
        
        allACC = np.hstack((allACC))
        mean_acc = np.round(np.mean(allACC), 2)
        std_acc = np.round(np.std(allACC), 2)
        mean_train_time = np.round(np.mean(allTrainTimes), 4)
        mean_pred_time = np.round(np.mean(allPredTimes), 4)
        
        accuracyList.append(mean_acc)
        
        print(f'Data partition: {partition}, Features: {feaCnt}, Accuracy: {mean_acc} +/- {std_acc}, '
              f'Mean training time: {mean_train_time} s, Mean prediction time: {mean_pred_time} s')
    
    return accuracyList


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
    fCntList = [10, 50]
    standardizeFlag = False
    preTrFlag = True

    # load training data data
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
    fCntList = [len(feaW)]
    # using the selected features run classification
    accuracyList = classifyALLAMLData(pp, feaList, fCntList, gpuId)
