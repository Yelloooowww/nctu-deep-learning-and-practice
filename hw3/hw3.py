from dataloader import read_bci_data
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR

class EEG(nn.Module):
	def __init__(self):
		super(EEG, self).__init__()
		self.pipe = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,51), stride=(1,1),padding=(0,25), bias=False),
			nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
			nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
			nn.ELU(32),
			nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
			nn.Dropout(p=0.25),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,15),stride=(1,1),padding=(0,7), bias=False),
			nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
			nn.ELU(32),
			nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
			nn.Dropout(p=0.25),
			nn.Flatten(),
			nn.Linear(in_features=736,out_features=2,bias=True)
		)
	def forward(self,x):
		x = self.pipe(x)
		return x


class DeepConvNet(nn.Module):
	def __init__(self):
		super(DeepConvNet, self).__init__()
		C, T, N = 2, 750, 2
		self.pipe = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,5)),
			nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(C,1)),
			nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
			nn.ELU(25),
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,5)),
			nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
			nn.ELU(50),
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1,5)),
			nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
			nn.ELU(100),
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1,5)),
			nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
			nn.ELU(200),
			nn.MaxPool2d(kernel_size=(1,2)),
			nn.Dropout(p=0.5),
			nn.Flatten(),
			nn.Linear(in_features=8600,out_features=2)
		)

	def forward(self,x):
		x = self.pipe(x)
		return x


i = 0
def train( model, train_data, train_label, optimizer):
	global i
	count = 0
	model.train()
	while count<1080:
		data = torch.cuda.FloatTensor( train_data[i:i+64] )
		target = torch.cuda.LongTensor( train_label[i:i+64] )
		optimizer.zero_grad()
		output = model(data)
		loss = nn.CrossEntropyLoss()
		loss = loss(output, target)
		loss.backward()
		optimizer.step()

		i = (i+64)%1080
		count += 64

def test(model, test_data, test_label, epoch):
	model.eval()
	data = torch.cuda.FloatTensor( test_data )
	target = torch.cuda.LongTensor( test_label )
	output = model(data)
	loss = nn.CrossEntropyLoss()
	test_loss = loss(output, target)  # sum up batch loss
	pred = output.argmax(dim=1)  # get the index of the max log-probability
	correct = 0
	for i,pred_ans in enumerate(pred):
		if pred[i] == target[i]: correct += 1
	print('epoch= ',epoch,'test_loss= ',test_loss.item()/1080.0,' correct= ',correct/1080.0)



if __name__ == '__main__':
	torch.manual_seed(1)
	device = torch.device('cuda:0')
	train_data, train_label, test_data, test_label = read_bci_data()

	model = EEG()
	model.to(device)
	optimizer = optim.Adam(model.parameters(),lr=0.0011)
	scheduler = StepLR(optimizer, step_size=100, gamma=0.999)
	for epoch in range(150):
		train(model, train_data, train_label, optimizer)
		test(model, test_data, test_label, epoch=epoch)
		scheduler.step()
	torch.save(model.state_dict(), "hw3_DeepConvNet.pt")


	model = DeepConvNet()
	model.to(device)
	optimizer = optim.Adam(model.parameters(),lr=0.0001)
	for epoch in range(150):
		train(model, train_data, train_label, optimizer)
		test(model, test_data, test_label, epoch=epoch)
	torch.save(model.state_dict(), "hw3_EEG.pt")
