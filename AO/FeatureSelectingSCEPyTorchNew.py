import pdb
from copy import deepcopy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
#from CustomizedLinear import CustomizedLinear
from OneToOneLinear import OneToOneLinear
from utilityScript import createOutputAsCentroids, standardizeData



class FeatureSelectingSCE(nn.Module):
	def __init__(self, netConfig={}):
		# 配置网络
		super(FeatureSelectingSCE, self).__init__()
		if len(netConfig.keys()) != 0:		
			self.inputDim,self.outputDim = netConfig['inputDim'],netConfig['inputDim']
			self.hLayer,self.hLayerPost = deepcopy(netConfig['hL']),deepcopy(netConfig['hL'])			
			self.l1Penalty,self.l2Penalty = 0.0,0.0
			self.weightCutoff = None
			self.oActFunc,self.errorFunc = 'linear','MSE'
			self.linearDecoder=False
			
			self.splWs = None
			self.fWeight,self.fIndices = [],[]
			#pdb.set_trace()
			if 'weightCutoff' in netConfig.keys(): self.weightCutoff = netConfig['weightCutoff']
			if 'l1Penalty' in netConfig.keys(): self.l1Penalty = netConfig['l1Penalty']
			if 'l2Penalty' in netConfig.keys(): self.l2Penalty = netConfig['l2Penalty']
			if 'errorFunc' in netConfig.keys(): self.errorFunc = netConfig['errorFunc']
			if 'oActFunc' in netConfig.keys(): self.oActFunc = netConfig['oActFunc']
			if 'linearDecoder' in netConfig.keys(): self.linearDecoder=netConfig['linearDecoder']
			
			self.hActFunc,self.hActFuncPost=deepcopy(netConfig['hActFunc']),deepcopy(netConfig['hActFunc'])

		else:#for default set up
			self.hLayer=[2]
			self.oActFunc,self.errorFunc='linear','MSE'
			self.hActFunc,self.hActFuncPost='tanh','tanh'

		#internal variables
		self.epochError=[]
		self.trMu=[]
		self.trSd=[]
		self.tmpPreHActFunc=[]
		self.preTrW,self.preTrB = [],[]
		self.runningPostTr = False
		self.device = None
		self.preTrItr = None

	def initNet(self,input_size,hidden_layer):
		# 网络初始化
		# 预训练阶段，标准线性层
		# 后续训练阶段，添加一个OneToOneLinear层，用于特征选择
		self.hidden=nn.ModuleList()
		# Hidden layers
		if not(self.runningPostTr):
			if len(hidden_layer)==1:
				self.hidden.append(nn.Linear(input_size,hidden_layer[0]))
				
			elif(len(hidden_layer)>1):
				for i in range(len(hidden_layer)-1):
					if i==0:
						self.hidden.append(nn.Linear(input_size, hidden_layer[i]))
						self.hidden.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
					else:
						self.hidden.append(nn.Linear(hidden_layer[i],hidden_layer[i+1]))
		else:
			#pdb.set_trace()
			for i in range(len(hidden_layer)-1):
				if i==0:
					self.hidden.append(OneToOneLinear(self.inputDim))
					self.hidden.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
				else:
					self.hidden.append(nn.Linear(hidden_layer[i],hidden_layer[i+1]))
		
		if not(self.runningPostTr):
			self.reset_parameters(hidden_layer)
		# Output layer
		self.out = nn.Linear(hidden_layer[-1], input_size)
		
	def reset_parameters(self,hidden_layer):
		# 网络参数初始化
		hL = 0
		while True:
			if self.hActFunc[hL].upper() in ['SIGMOID','TANH']:
				torch.nn.init.xavier_uniform_(self.hidden[hL].weight)
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			elif self.hActFunc[hL].upper() == 'RELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			elif self.hActFunc[hL].upper() == 'LRELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='leaky_relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			if hL == len(hidden_layer)-1:
				break
			hL += 1

	def forwardPost(self, x):
		# 后训练阶段前向传播，包含稀疏层的传播
		# Feedforward		
		for l in range(len(self.hidden)):
			if self.hActFuncPost[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='RELU':
				x = F.relu(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
			else:#default is linear				
				x = self.hidden[l](x)

		if self.oActFunc.upper()=='SIGMOID':
			return torch.sigmoid(self.out(x))
		else:
			return self.out(x)

	def forwardPre(self, x):
		# 预训练阶段前向传播，标准线性层的传播
		# Feedforward
		if self.tmpPreHActFunc[0].upper()=='SIGMOID':
			x = torch.sigmoid(self.hidden[1](x))
		elif self.tmpPreHActFunc[0].upper()=='TANH':
			x = torch.tanh(self.hidden[1](x))
		elif self.tmpPreHActFunc[0].upper()=='RELU':
			x = torch.relu(self.hidden[1](x))
		elif self.tmpPreHActFunc[0].upper()=='LRELU':
			x = F.leaky_relu(self.hidden[1](x),inplace=False)
		else:#default is linear
			x = self.hidden[1](x)
		if self.oActFunc.upper()=='SIGMOID':
			return torch.sigmoid(self.out(x))
		else:
			return self.out(x)

	def setHiddenWeight(self,W,b):
		for i in range(len(self.hidden)):
			self.hidden[i].bias.data=b[i].float()
			self.hidden[i].weight.data=W[i].float()

	def setOutputWeight(self,W,b):
		self.out.bias.data=b.float()
		self.out.weight.data=W.float()

	def returnTransformedData(self,x):
		fOut=[x]
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			for layer in self.hidden:
				fOut.append(self.hiddenActivation(layer(fOut[-1])))
			if self.output_activation.upper()=='SIGMOID':
				fOut.append(torch.sigmoid(self.out(fOut[-1])))
			else:
				fOut.append(self.out(fOut[-1]))
		return fOut[1:]#Ignore the original input
		
	def Train(self,dataLoader,optimizationFunc,learningRate,m,batchSize,numEpochs,verbose):
		self.runningPostTr = True
		self.hLayerPost = np.insert(self.hLayerPost,0,self.inputDim)
		self.hActFuncPost.insert(0,'SPL')
		self.initNet(self.inputDim,self.hLayerPost)
		optimizer = torch.optim.Adam(self.parameters(),lr=learningRate,amsgrad=True)
		criterion = nn.MSELoss()
		for epoch in range(numEpochs):
			self.tmpPreHActFunc=self.hActFunc[0]
			self.hidden[0].weight.requires_grad = False
			self.hidden[1].weight.requires_grad = True
			self.to(self.device)
			for trInput, trOutput in dataLoader:
				trInput, trOutput = trInput.to(self.device), trOutput.to(self.device)
				optimizer.zero_grad()
				outputs = self.forwardPre(trInput)
				loss = criterion(outputs, trOutput)
				loss.backward()
				optimizer.step()
				
			self.hidden[0].weight.requires_grad = True
			self.hidden[1].weight.requires_grad = False
			self.to(self.device)
			for trInput, trOutput in dataLoader:
				trInput, trOutput = trInput.to(self.device), trOutput.to(self.device)
				optimizer.zero_grad()
				outputs = self.forwardPost(trInput)
				loss = criterion(outputs, trOutput)
				if self.l1Penalty != 0: 
					l1_loss = torch.norm(self.hidden[0].weight, p=1)
					loss += self.l1Penalty * l1_loss
				loss.backward()
				optimizer.step()
			if verbose:
				print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item()}, L1 Norm of SPL: {torch.sum(torch.abs(self.hidden[0].weight)).item()}")
		self.splWs = self.hidden[0].weight.data.to('cpu')
		self.fWeight,self.fIndices = torch.sort(torch.abs(self.splWs),descending=True)
		self.fWeight = self.fWeight.to('cpu').numpy()
		self.fIndices = self.fIndices.to('cpu').numpy()

	def fit(self,trData,trLabels,valData=[],valLabels=[],preTraining=True,optimizationFunc='Adam',learningRate=0.001,m=0,
			miniBatchSize=100,numEpochsPreTrn=25,numEpochsPostTrn=100,standardizeFlag=True,cudaDeviceId=0,verbose=True):

		# set device
		self.device = torch.device('cuda:'+str(cudaDeviceId))

		#standardize data
		if standardizeFlag:
			mu,sd,trData = standardizeData(trData)
			#_,_,target = standardizeData(target)
			self.trMu=mu
			self.trSd=sd

		#create target: centroid for each class
		target=createOutputAsCentroids(trData,trLabels)
		#pdb.set_trace()

		#create target of validation data for early stopping
		#if len(valData) != 0:
		#	valData = standardizeData(valData,self.trMu,self.trSd)
		

		#Prepare data for torch
		trDataTorch=Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(target).float())
		dataLoader=Data.DataLoader(dataset=trDataTorch,batch_size=miniBatchSize,shuffle=True)
		self.Train(dataLoader,optimizationFunc,learningRate,m,miniBatchSize,numEpochsPostTrn,verbose)
		'''#layer-wise pre-training
		#pdb.set_trace()
		if preTraining:
			self.preTrain(dataLoader,optimizationFunc,learningRate,m,miniBatchSize,numEpochsPreTrn,verbose)
		else:
			#initialize the network weight and bias
			self.initNet(self.inputDim,self.hLayerPost)
		#post training
		self.postTrain(dataLoader,optimizationFunc,learningRate,m,miniBatchSize,numEpochsPostTrn,verbose)'''
		
	def predict(self,x):
		if len(self.trMu) != 0 and len(self.trSd) != 0:#standarization has been applied on training data so apply on test data
			x = standardizeData(x,self.trMu,self.trSd)
		#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		x=torch.from_numpy(x).float().to(device)
		fOut=[x]
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			for l in range(len(self.hidden)):
				if self.hActFuncPost[l].upper()=='SIGMOID':
					fOut.append(torch.sigmoid(self.hidden[l](fOut[-1])))
				elif self.hActFuncPost[l].upper()=='TANH':
					fOut.append(torch.tanh(self.hidden[l](fOut[-1])))
				elif self.hActFuncPost[l].upper()=='RELU':
					fOut.append(torch.relu(self.hidden[l](fOut[-1])))
				else:#default is linear				
					fOut.append(self.hidden[l](fOut[-1]))

			if self.oActFunc.upper()=='SIGMOID':
				fOut.append(torch.sigmoid(self.out(fOut[-1])))
			else:
				fOut.append(self.out(fOut[-1]))

		return fOut
