from zipfile import ZipFile
import os
import numpy as np
import pickle

def getApplicationData(dataSetName,partition):
# 加载数据集
	if dataSetName.upper()=='ALLAML':
		
		if not(os.path.isdir('ALLAML')):  # files were not unzipped
			files = ['./Data/ALLAML.zip']
			with ZipFile('./Data/ALLAML.zip', 'r') as zip:
				zip.extractall()
		trnSet = pickle.load(open('./ALLAML/Partition'+str(partition)+'/trnData.p','rb'))
		tstSet = pickle.load(open('./ALLAML/Partition'+str(partition)+'/tstData.p','rb'))

	elif dataSetName.upper()=='ISOLET':
		if not(os.path.isdir('ISOLET')):  # files were not unzipped
			with ZipFile('./Data/ISOLET.zip', 'r') as zip:
				zip.extractall()
		trnSet = pickle.load(open('./ISOLET/Isolet_Trn.p','rb'))
		tstSet = pickle.load(open('./ISOLET/Isolet_Tst.p','rb'))

	elif dataSetName.upper()=='COIL20':
		
		if not(os.path.isdir('COIL20')):  # files were not unzipped
			with ZipFile('./Data/COIL20.zip', 'r') as zip:
				zip.extractall()
		trnSet = pickle.load(open('./COIL20/COIL20_Trn.p','rb'))
		tstSet = pickle.load(open('./COIL20/COIL20_Tst.p','rb'))

	return trnSet,tstSet

def calcCentroid(data,label):
	centroids=[]
	centroidLabels=np.unique(label)
	for i in range(len(centroidLabels)):
		tmpData=data[np.where(centroidLabels[i]==label)[0],:]
		centroids.append(np.mean(tmpData,axis=0))
	centroids=np.vstack((centroids))
	return centroids

def createOutputAsCentroids(data, labels):
	# 生成每个类别的中心点
    """
    For each sample, return the centroid of its corresponding class.
    该函数针对每个样本，根据其 label 返回所属类别在 data 中的 centroid，
    从而构造出一个行数与 data 相同的 target 矩阵。
    """
    unique_labels = np.unique(labels)
    centroids = {}
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        centroids[lab] = np.mean(data[idx, :], axis=0)
    # 为每个样本分配对应类别的 centroid
    target = np.array([centroids[l] for l in labels])
    return target

def returnImpFeaturesElbow(W):
	# 
	#This program will use the concept of elbow search
	
	#sort the weights in descending order on the absolute values
	
	sortedW = (-1)*np.sort((-1)*np.abs(W))
	sortedIndices = np.argsort((-1)*np.abs(W))
	
	#assign each ordered weight value as a point in xy plane, where y values are the weights and x values are number in R(+)
	X = np.arange(len(sortedW))
	P = np.hstack((X.reshape(-1,1),sortedW.reshape(-1,1)))
	
	#now using the first and last point make a straight line y = mx + c or mx - y + c = 0
	m = (sortedW[0] - sortedW[-1])/(X[0] - X[-1])
	c = sortedW[0] - m * X[0]
	
	#now calculate the distance for each point in P from the straight line.
	# the elbow point will be the point in P whihc has maximum distance
	#The distance of a point P(x_o,y_0) from a straight line Ax + By + C = 0 is given by:
	# d = |(Ax_0 + By_0 + C)|/sqrt(A^2 + B^2)
	#In our case A = m, B = -1 and C = c
	dists = []
	denom = np.sqrt(1+m**2)
	for p in P:
		x_0,y_0 = p[0],p[1]
		numerator = np.abs(m*x_0 + (-1)*y_0 + c)
		dists.append(numerator/denom)
	dists = np.array(dists)
	maxDistIndex = np.where(dists==max(dists))[0][0]
	return sortedIndices[:maxDistIndex+1],sortedW[:maxDistIndex+1]

def standardizeData(data,mu=[],std=[]):
	#data: a m x n matrix where m is the no of observations and n is no of features
	#if any(mu) == None and any(std) == None:
	if not(len(mu) and len(std)):
		
		std = np.std(data,axis=0)
		mu = np.mean(data,axis=0)
		std[np.where(std==0)[0]] = 1.0 #This is for the constant features.
		standardizeData = (data - mu)/std
		return mu,std,standardizeData
	else:
		standardizeData = (data - mu)/std
		return standardizeData
		
def unStandardizeData(data,mu,std):
	return std * data + mu
