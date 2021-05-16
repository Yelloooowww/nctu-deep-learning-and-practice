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
import copy
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
	def __init__(self, train=True, shuffle=True):
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

		self.tenses = [0,1,2,3] #['simple-present','third-person','present-progressive','simple-past']
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

################################models###################################
#Encoder
class EncoderRNN(nn.Module):
	def __init__(
		self, word_size, hidden_size, latent_size,
		num_condition, condition_size
	):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.cell_size = hidden_size
		self.condition_size = condition_size
		self.latent_size = latent_size
		self.condition_embedding = nn.Embedding(num_condition, condition_size)
		self.word_embedding = nn.Embedding(word_size, hidden_size)
		self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size)
		self.mean = nn.Linear(hidden_size, latent_size)
		self.logvar = nn.Linear(hidden_size, latent_size)
		print('EncoderRNN init done')


	def forward(self, inputs, c, encoder_in_hidden=None, encoder_in_cell=None):
		input_seq = self.word_embedding(inputs.to(device)).view(-1, 1, self.hidden_size)
		c = self.condition(c) #torch.Size([1, 1, 8])
		hidden = torch.zeros(1, 1, self.hidden_size-self.condition_size ,device=device)
		hidden = torch.cat((hidden, c), dim=2)
		cell = torch.zeros(1, 1, self.hidden_size-self.condition_size ,device=device)
		cell = torch.cat((cell, c), dim=2)
		out, (h_n, c_n) = self.lstm(input_seq,(hidden,cell))
		mean = self.mean(h_n) # Latent torch.Size([1, 1, 32])
		logvar = self.logvar(h_n) # Latent torch.Size([1, 1, 32])
		z = (torch.randn(1,1,self.latent_size).to(device)) * torch.exp(logvar/2) + mean # Latent torch.Size([1, 1, 32])
		return out, h_n, c_n, z, mean, logvar # z:latent


	def condition(self, c):
		c = torch.LongTensor([c]).to(device)
		return self.condition_embedding(c).view(1,1,-1)

#Decoder
class DecoderRNN(nn.Module):
	def __init__(self, word_size, hidden_size, latent_size, condition_size):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.word_size = word_size
		self.condition_size = condition_size
		self.latent_to_hidden = nn.Linear(latent_size+condition_size, hidden_size)
		self.word_embedding = nn.Embedding(word_size, hidden_size)
		self.condition_embedding = nn.Embedding(num_condition, condition_size)
		self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size)
		self.out = nn.Linear(hidden_size, word_size)
		self.softmax = nn.LogSoftmax(dim=1)
		self.bn = nn.BatchNorm1d(num_features=word_size)
		print('DecoderRNN init done')

	def initCell(self):
		return torch.zeros(1, 1, self.hidden_size - self.condition_size,device=device)

	def forward(self, input_seq, hidden, cell):
		out, (h_n, c_n) = self.lstm(input_seq, (hidden, cell) )
		output = self.out(out[0])
		# output = self.softmax(output)
		return output.to(device), out, (h_n, c_n) # output-> prob. distrubution ; out -> latant

	def condition(self, c):
		c = torch.LongTensor([c]).to(device)
		return self.condition_embedding(c).view(1,1,-1)

#############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataset = wordsDataset(shuffle=True)
test_dataset = wordsDataset(False,shuffle=True)
word_size = train_dataset.chardict.n_words #26+2
num_condition = len(train_dataset.tenses) # 4 tenses
hidden_size = 256
latent_size = 32
condition_size = 8
teacher_forcing_ratio = 0.5
KLD_weight = 0.5
LR = 0.05
epochs = 1000
KL_cost_annealing = 'cyclical' # or 'monotonic'

encoder = EncoderRNN(word_size, hidden_size, latent_size, num_condition, condition_size).to(device)
decoder = DecoderRNN(word_size, hidden_size, latent_size, condition_size).to(device)
assert encoder.hidden_size == decoder.hidden_size, \
	"Hidden dimensions of encoder and decoder must be equal!"

encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
###############################################################################


def use_decoder(z, c, inputs, use_teacher_forcing):

	sos_token = train_dataset.chardict.word2index['SOS'] # sos_token = 26
	eos_token = train_dataset.chardict.word2index['EOS'] # eos_token = 27
	outputs = []
	if not inputs == None:
		maxlen = inputs.size(0)
	else :
		maxlen = 16

	de_input = torch.LongTensor([sos_token]).to(device)
	de_input_seq = decoder.word_embedding(de_input.to(device)).view(-1, 1, decoder.hidden_size)
	c = decoder.condition(c)
	latent = torch.cat((z, c), dim=2)
	de_hidden = decoder.latent_to_hidden(latent)
	de_cell = decoder.initCell()
	de_cell = torch.cat((de_cell, c), dim=2)

	entropy_loss = 0
	for i in range(1,maxlen):
		output, out, (de_hidden, de_cell) = decoder(de_input_seq, de_hidden, de_cell )
		out_onehot = torch.max(torch.softmax(output, dim=1), 1)[1] # [1]只返回最大值的每个索引
		outputs.append(output)
		if not inputs == None: entropy_loss += criterion(output, torch.tensor([inputs[i]]).to(device))
		if out_onehot.item() == eos_token and not use_teacher_forcing: break
		if use_teacher_forcing:
			de_input = inputs[i]
			de_input_seq = decoder.word_embedding(de_input.to(device)).view(-1, 1, decoder.hidden_size)
		else:
			de_input_seq = out



	if len(outputs) != 0:
		outputs = torch.cat(outputs, dim=0)
	else:
		outputs = torch.FloatTensor([]).view(0, word_size).to(device)

	return outputs, entropy_loss/(maxlen-1)
	# return outputs, entropy_loss

def KL_loss(m, logvar):
	# return -0.5 * torch.sum(1 + logvar - m**2 - logvar.exp())
	return 0.5 * torch.sum(torch.exp(logvar) + m**2 - 1. - logvar)


def evaluation(encoder, decoder, dataset,show=True):
	blue_score = []
	for idx in range(len(dataset)):
		data = dataset[idx]
		if dataset.train:
			inputs, c = data # inputs-> word(tensor([26,  0,  1,  0, 13,  3, 14, 13, 27])) ,
							 # c-> tense conversion (0)
			targets = inputs
			target_condition = c
		else:
			inputs, c, targets, target_condition = data

		_, _, _, encoder_out_hidden, _, _ = encoder(targets, target_condition)
		outputs,_ = use_decoder(encoder_out_hidden, target_condition, targets, use_teacher_forcing=False)

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

	words_list = []
	for j in range(100):
		words = []
		noise = torch.randn(1,1,latent_size).to(device)
		for i in range(len(train_dataset.tenses)):
			outputs,_ = use_decoder(noise, int(i), None, False)
			outputs_onehot = torch.max(torch.softmax(outputs, dim=1), 1)[1]
			output_str = train_dataset.chardict.stringFromLongtensor(outputs_onehot)
			words.append(output_str)
		words_list.append(words)
	gaussian_score = Gaussian_score(words)
	if show:
		print('generate word: ')
		for i in range(len(words_list)): print(words_list[i])
		print(' gaussian_score:',Gaussian_score(words_list))

	return sum(blue_score) / len(blue_score) , gaussian_score


#compute BLEU-4 score
def compute_bleu(output, reference):
	cc = SmoothingFunction()
	if len(reference) == 3:
		weights = (0.33,0.33,0.33)
	else:
		weights = (0.25,0.25,0.25,0.25)
	return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""

def Gaussian_score(words):
	words_list = []
	score = 0
	yourpath = 'dataset/train.txt'#should be your directory of train.txt
	with open(yourpath,'r') as fp:
		for line in fp:
			word = line.split(' ')
			word[3] = word[3].strip('\n')
			words_list.extend([word])
		for t in words:
			for i in words_list:
				if t == i:
					score += 1
	return score/len(words)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def get_kl_weight(epoch,epochs,kl_annealing_type,time):
	"""
	:param epoch: i-th epoch
	:param kl_annealing_type: 'monotonic' or 'cycle'
	:param time:
		if('monotonic'): # of epoch for kl_weight from 0.0 to reach 1.0
		if('cycle'):     # of cycle
	"""
	assert kl_annealing_type=='monotonic' or kl_annealing_type=='cycle','kl_annealing_type not exist!'

	if kl_annealing_type == 'monotonic':
		return (1./(time-1))*(epoch-1) if epoch<time else 1.

	else: #cycle
		period = epochs//time
		epoch %= period
		KL_weight = sigmoid((epoch - period // 2) / (period // 10)) / 2
		return KL_weight



if __name__=='__main__':
	#plot list
	tfr_list, kld_w_list, LR_list = [], [], []
	all_loss, crossentropy_list, KL_loss_list, bleu_list, gaussian_list = [], [], [], [], []

	best_bleu = 0
	# NEW
	# scaler = torch.cuda.amp.GradScaler()
	for epoch in range(1, epochs+1):

		# if KL_cost_annealing == 'cyclical':
		# 	tmp = epoch%10
		# 	if tmp < 5 : kld_w = 4
		# 	else: kld_w = tmp*0.8
		# if KL_cost_annealing == 'monotonic':
		# 	if epoch < epochs*0.5 : kld_w = 1
		# 	else : kld_w = 1-(epoch-epochs*0.5)/(epochs*0.5)
		kld_w = get_kl_weight(epoch,epochs,'cycle',2)


		# tfr = 1.-(1./epochs)*epoch if epoch % 2 ==0  else 0
		tfr = 1.-(1./epochs)*epoch
		# if epoch < epochs*0.1 : tfr = 0.9
		# else :
		# 	tfr = max(0.9-(1./epochs)*epoch, 0)

		encoder.train()
		decoder.train()
		for i in range(len(train_dataset)):
			idx = random.randint(0, len(train_dataset)-1)

			crossentropy_list_tmp, KL_loss_list_tmp = [], []
			data = train_dataset[idx]
			inputs, c = data # inputs-> word(tensor([26,  0,  1,  0, 13,  3, 14, 13, 27])) ,
							 # c-> tense conversion (0)

			# # NEW
			# with torch.cuda.amp.autocast():
			decoder_optimizer.zero_grad()
			encoder_optimizer.zero_grad()
			_, _, _, z, mean, logvar = encoder(inputs, c)
			use = True if random.random() < tfr else False
			outputs, entropy_loss = use_decoder(z, c, inputs, use_teacher_forcing=use) #inputs = ground truth
			#
			# outputs_onehot = torch.max(torch.softmax(outputs, dim=1), 1)[1]
			# inputs_str = train_dataset.chardict.stringFromLongtensor(inputs, check_end=True)
			# outputs_str = train_dataset.chardict.stringFromLongtensor(outputs_onehot, check_end=True)
			# print('inputs_str: ',inputs_str,'  outputs_str: ',outputs_str)


			output_length = outputs.size(0)
			# CrossEntropyLoss Input: (N, C)(N,C) where C = number of classes
			# CrossEntropyLoss Target: (N)
			# entropy_loss = criterion(outputs, inputs[1:1+output_length].to(device))

			kld_loss = KL_loss(mean, logvar)
			# # NEW
			# scaler.scale( (entropy_loss + (kld_w * kld_loss)) ).backward()
			(entropy_loss + (kld_w * kld_loss)).backward()
			# NEW
			# scaler.step(encoder_optimizer)
			# scaler.step(decoder_optimizer)
			# scaler.update()


			encoder_optimizer.step()
			decoder_optimizer.step()
			crossentropy_list_tmp.append(entropy_loss.item())
			KL_loss_list_tmp.append(kld_loss.item())

		encoder.eval()
		decoder.eval()
		blue_score, gaussian_score = evaluation(encoder, decoder, test_dataset,show=True)


		#plot
		tfr_list.append(tfr)
		kld_w_list.append(kld_w)
		LR_list.append(LR)
		crossentropy_list.append(sum(crossentropy_list_tmp) / len(crossentropy_list_tmp))
		KL_loss_list.append(sum(KL_loss_list_tmp) / len(KL_loss_list_tmp))
		bleu_list.append(blue_score)
		gaussian_list.append(gaussian_score)
		all_loss.append( (crossentropy_list[-1] + (kld_w * KL_loss_list[-1])) )
		print('epoch= ',epoch, 'blue_score=',blue_score, 'gaussian_score=',gaussian_score, 'crossentropy=',crossentropy_list[-1], 'KL_loss=',KL_loss_list[-1])
		print('all loss=', all_loss[-1])

		# save
		if blue_score > best_bleu:
			best_encoder_wts = copy.deepcopy(encoder.state_dict())
			torch.save(best_encoder_wts,f'models/best_encoder_epoch_{epoch}_bleu_{blue_score}.pt')
			best_decoder_wts = copy.deepcopy(decoder.state_dict())
			torch.save(best_decoder_wts,f'models/best_decoder_epoch_{epoch}_bleu_{blue_score}.pt')
			best_bleu = blue_score




	fig = plt.figure()
	plt.plot(tfr_list,color='b')
	plt.plot(kld_w_list,color='g')
	plt.plot(LR_list,color='r')
	plt.plot(crossentropy_list,color='c')
	plt.plot(KL_loss_list,color='m')
	plt.plot(bleu_list,color='y')
	plt.plot(gaussian_list,color='k')
	plt.plot(all_loss,color='lime')
	plt.xlabel('epoch')
	plt.show()
	fig.savefig('BLEU'+'Crossentropyloss'+'KLloss'+'.png')
