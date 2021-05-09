from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import os
import sys
import math
import json
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu



###############################loader###################################
class CharDict:
	def __init__(self):
		self.word2index = {}
		self.index2word = {}
		self.n_words = 0
		for i in range(26): self.addWord(chr(ord('a') + i))
		tokens = ["SOS", "EOS"]
		for t in tokens: self.addWord(t)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.index2word[self.n_words] = word
			self.n_words += 1

	def longtensorFromString(self, s):
		s = ["SOS"] + list(s) + ["EOS"]
		return torch.LongTensor([self.word2index[ch] for ch in s])

	def stringFromLongtensor(self, l, show_token=False, check_end=True):
		s = ""
		for i in l:
			ch = self.index2word[i.item()]
			if len(ch) > 1:
				if show_token: __ch = "<{}>".format(ch)
				else: __ch = ""
			else: __ch = ch
			s += __ch
			if check_end and ch == "EOS": break
		return s

class wordsDataset(Dataset):
	def __init__(self, train=True):
		if train: f = 'dataset/train.txt'
		else: f = 'dataset/test.txt'
		self.datas = np.loadtxt(f, dtype=np.str)

		if train: self.datas = self.datas.reshape(-1)
		else:
			self.targets = np.array([
				[0, 3],  # sp -> p
				[0, 2],  # sp -> pg
				[0, 1],  # sp -> tp
				[0, 1],  # sp -> tp
				[3, 1],  # p  -> tp
				[0, 2],  # sp -> pg
				[3, 0],  # p  -> sp
				[2, 0],  # pg -> sp
				[2, 3],  # pg -> p
				[2, 1],  # pg -> tp
			])

		self.tenses = ['simple-present','third-person','present-progressive','simple-past']
		self.chardict = CharDict()
		self.train = train

	def __len__(self):
		return len(self.datas)

	def __getitem__(self, index):
		if self.train:
			c = index % len(self.tenses)
			return self.chardict.longtensorFromString(self.datas[index]), c
		else:
			i = self.chardict.longtensorFromString(self.datas[index, 0])
			ci = self.targets[index, 0]
			o = self.chardict.longtensorFromString(self.datas[index, 1])
			co = self.targets[index, 1]
			return i, ci, o, co

def decode_inference(decoder, z, c, maxlen, teacher=False, inputs=None):
	sos_token = train_dataset.chardict.word2index['SOS']
	eos_token = train_dataset.chardict.word2index['EOS']
	z = z.view(1,1,-1)
	i = 0

	outputs = []
	x = torch.LongTensor([sos_token]).to(device)
	hidden = decoder.initHidden(z, c)

	for i in range(maxlen):
		# get (1, word_size), (1,1,hidden_size)
		x = x.detach()
		output, hidden = decoder(x, hidden)
		outputs.append(output)
		output_onehot = torch.max(torch.softmax(output, dim=1), 1)[1]
		# meet EOS
		if output_onehot.item() == eos_token and not teacher: break
		if teacher: x = inputs[i+1:i+2]
		else: x = output_onehot

	# get (seq, word_size)
	if len(outputs) != 0: outputs = torch.cat(outputs, dim=0)
	else: outputs = torch.FloatTensor([]).view(0, word_size).to(device)

	return outputs

################################models###################################
#Encoder
class EncoderRNN(nn.Module):
	def __init__(
		self, word_size, hidden_size, latent_size,
		num_condition, condition_size
	):
		super(EncoderRNN, self).__init__()
		self.word_size = word_size
		self.hidden_size = hidden_size
		self.condition_size = condition_size
		self.latent_size = latent_size

		self.condition_embedding = nn.Embedding(num_condition, condition_size)
		self.word_embedding = nn.Embedding(word_size, hidden_size)
		self.lstm = nn.LSTM(hidden_size, hidden_size)
		self.mean = nn.Linear(hidden_size, latent_size)
		self.logvar = nn.Linear(hidden_size, latent_size)

	def forward(self, inputs, init_hidden, input_condition):
		c = self.condition(input_condition)
		hidden = torch.cat((init_hidden, c), dim=2)# get (1,1,hidden_size)
		x = self.word_embedding(inputs).view(-1, 1, self.hidden_size)# get (seq, 1, hidden_size)
		out, (h_n, c_n) = self.lstm(x)
		m = self.mean(h_n)
		logvar = self.logvar(h_n)
		z = self.sample_z() * torch.exp(logvar/2) + m
		return z, m, logvar

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size - self.condition_size,device=device)

	def condition(self, c):
		c = torch.LongTensor([c]).to(device)
		return self.condition_embedding(c).view(1,1,-1)

	def sample_z(self):
		return torch.normal(
			torch.FloatTensor([0]*self.latent_size),
			torch.FloatTensor([1]*self.latent_size)
		).to(device)

#Decoder
class DecoderRNN(nn.Module):
	def __init__(self, word_size, hidden_size, latent_size, condition_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.word_size = word_size
		self.latent_to_hidden = nn.Linear(latent_size+condition_size, hidden_size)
		self.word_embedding = nn.Embedding(word_size, hidden_size)
		self.lstm = nn.LSTM(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, word_size)

	def initHidden(self, z, c):
		latent = torch.cat((z, c), dim=2)
		return self.latent_to_hidden(latent)

	def forward(self, x, hidden):
		output = self.word_embedding(x).view(1, 1, -1)
		# output = F.relu(output)
		output, (h_n, c_n) = self.lstm(output)
		output = self.out(output).view(-1, self.word_size)# get (1, word_size)
		return output, hidden

#############################################################################
def KLD_weight_annealing(epoch):
	# slope = 0.001
	slope = 0.1
	scope = (1.0 / slope)*2
	w = (epoch % scope) * slope
	if w > 1.0: w = 1.0
	return w

def teacher_forcing_ratio(epoch):
	print('teacher_forcing_ratio')
	# from 1.0 to 0.0
	slope = 0.01
	level = 10
	w = 1.0 - (slope * (epoch//level))
	if w <= 0.0: w = 0.0
	return w

def KL_loss(m, logvar):
	return torch.sum(0.5 * (-logvar + (m**2) + torch.exp(logvar) - 1))

def generate_word(encoder, decoder, z, condition, maxlen=20):
	encoder.eval()
	decoder.eval()
	outputs = decode_inference(decoder, z, encoder.condition(condition), maxlen=maxlen)
	return torch.max(torch.softmax(outputs, dim=1), 1)[1]

def compute_bleu(output, reference):
	cc = SmoothingFunction()
	return sentence_bleu(
		[reference], output,
		weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=cc.method1
	)

def evaluation(encoder, decoder, dataset,show=True):
	encoder.eval()
	decoder.eval()

	blue_score = []

	for idx in range(len(dataset)):
		data = dataset[idx]
		if dataset.train:
			inputs, input_condition = data
			targets = inputs
			target_condition = input_condition
		else:
			inputs, input_condition, targets, target_condition = data

		# input no sos and eos
		z, _, _ = encoder(inputs[1:-1].to(device), encoder.initHidden(), input_condition)

		# input has sos and eos

		outputs = decode_inference(decoder, z, encoder.condition(target_condition), maxlen=len(targets))

		# show output by string
		outputs_onehot = torch.max(torch.softmax(outputs, dim=1), 1)[1]
		inputs_str = train_dataset.chardict.stringFromLongtensor(inputs, check_end=True)
		targets_str = train_dataset.chardict.stringFromLongtensor(targets, check_end=True)
		outputs_str = train_dataset.chardict.stringFromLongtensor(outputs_onehot, check_end=True)

		if show:
			print('input:',inputs_str)
			print('target:',targets_str)
			print('prediction:',outputs_str)
			print('')

		blue_score.append( compute_bleu(outputs_str, targets_str) )

	if show: print('BLEU-4 score : {}'.format(sum(blue_score) / len(blue_score)))
	if show:
		noise = encoder.sample_z()
		for i in range(len(train_dataset.tenses)):
			outputs = generate_word(encoder, decoder, noise, i)
			output_str = train_dataset.chardict.stringFromLongtensor(outputs)
			print(output_str,end=' ')
		print('')
	return blue_score





if __name__=='__main__':
	print('--- start ---')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# config
	train_dataset = wordsDataset()
	test_dataset = wordsDataset(False)

	word_size = train_dataset.chardict.n_words
	num_condition = len(train_dataset.tenses)
	hidden_size = 256
	latent_size = 32
	condition_size = 8

	teacher_forcing_ratio = 0.5
	empty_input_ratio = 0.1
	KLD_weight = 0.0
	LR = 0.05
	epoch_size = 500



	encoder = EncoderRNN(word_size, hidden_size, latent_size, num_condition, condition_size).to(device)
	decoder = DecoderRNN(word_size, hidden_size, latent_size, condition_size).to(device)
	encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR)
	criterion = nn.CrossEntropyLoss(reduction='sum')


	# for plot
	Crossentropyloss, KLloss, BLEU = [], [], []

	#epoch
	for epoch in range(1,epoch_size+1):
		encoder.train()
		decoder.train()
		for idx in range(len(train_dataset)):
			Crossentropyloss_list, KLloss_list = [], []


			data = train_dataset[idx]
			inputs, c = data
			encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()

			# input no sos and eos
			z, m, logvar = encoder(inputs[1:-1].to(device), encoder.initHidden(), c)

			if callable(teacher_forcing_ratio): tfr = teacher_forcing_ratio(epoch)
			else: tfr = teacher_forcing_ratio
			use_teacher_forcing = True if random.random() < tfr else False
			outputs = decode_inference(
				decoder, z, encoder.condition(c), maxlen=inputs[1:].size(0),
				teacher=tfr, inputs=inputs.to(device))

			# target no sos
			output_length = outputs.size(0)

			loss = criterion(outputs, inputs[1:1+output_length].to(device))
			Crossentropyloss_list.append(loss.item())

			kld_loss = KL_loss(m, logvar)
			KLloss_list.append(kld_loss.item())
			if callable(KLD_weight_annealing): kld_w = KLD_weight_annealing(epoch)
			else: kld_w = KLD_weight
			(loss + (kld_w * kld_loss)).backward()

			encoder_optimizer.step()
			decoder_optimizer.step()

		show_flag = True if epoch%50==0 else False
		# evaluation
		all_score = evaluation(encoder, decoder, test_dataset, show=show_flag)
		BLEU.append(np.mean(all_score))
		Crossentropyloss.append(np.mean(Crossentropyloss_list))
		KLloss.append(np.mean(KLloss_list))
		print('epoch:',epoch,BLEU[-1],Crossentropyloss[-1],KLloss[-1])



	fig = plt.figure()
	bleeeu = plt.plot(BLEU,color='r')
	klloss = plt.plot(KLloss,color='b')
	ax = plt.gca().twinx()
	closs = ax.plot(Crossentropyloss,color='g')
	plt.xlabel('epoch')
	# ax.legend([bleeeu,klloss,closs],[BLEU,KLloss,Crossentropyloss])
	plt.show()
	fig.savefig('BLEU'+'Crossentropyloss'+'KLloss'+'.png')
