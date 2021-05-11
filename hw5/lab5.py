from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import os
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
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.mean = nn.Linear(hidden_size, latent_size)
		self.logvar = nn.Linear(hidden_size, latent_size)

	def forward(self, inputs, init_hidden, input_condition):
		c = self.condition(input_condition)

		# get (1,1,hidden_size)
		hidden = torch.cat((init_hidden, c), dim=2)

		# get (seq, 1, hidden_size)
		x = self.word_embedding(inputs).view(-1, 1, self.hidden_size)

		# get (seq, 1, hidden_size), (1, 1, hidden_size)
		outputs, hidden = self.gru(x, hidden)

		# get (1, 1, hidden_size)
		m = self.mean(hidden)
		logvar = self.logvar(hidden)

		z = self.sample_z() * torch.exp(logvar/2) + m

		#self.m = m
		#self.logvar = logvar

		return z, m, logvar

	def initHidden(self):
		return torch.zeros(
			1, 1, self.hidden_size - self.condition_size,
			device=device
		)

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
	def __init__(
		self, word_size, hidden_size, latent_size, condition_size
	):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.word_size = word_size

		self.latent_to_hidden = nn.Linear(
			latent_size+condition_size, hidden_size
		)
		self.word_embedding = nn.Embedding(word_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)
		self.out = nn.Linear(hidden_size, word_size)

	def initHidden(self, z, c):
		latent = torch.cat((z, c), dim=2)
		return self.latent_to_hidden(latent)

	def forward(self, x, hidden):
		# get (1, 1, hidden_size)
		x = self.word_embedding(x).view(1, 1, self.hidden_size)

		# get (1, 1, hidden_size) (1, 1, hidden_size)
		output, hidden = self.gru(x, hidden)

		# get (1, word_size)
		output = self.out(output).view(-1, self.word_size)

		return output, hidden

	def forwardv1(self, inputs, z, c, teacher=False, hidden=None):
		# get (1,1,latent_size + condition_size)
		latent = torch.cat((z, c), dim=2)

		# get (1,1,hidden_size)
		if hidden is None:
			hidden = self.latent_to_hidden(latent)
			#print("get hidden from latent")

		# get (seq, 1, hidden_size)
		x = self.word_embedding(inputs).view(-1, 1, self.hidden_size)

		input_length = x.size(0)

		# get (seq, 1, hidden_size), (1, 1, hidden_size)
		if teacher:
			outputs = []
			for i in range(input_length-1):
				output, hidden = self.gru(x[i:i+1], hidden)
				hidden = x[i+1:i+2]
				outputs.append(output)

			outputs = torch.cat(outputs, dim=0)
		else:
			# Omit EOS token
			x = x[:-1]
			outputs, hidden = self.gru(x, hidden)


		# get (seq, word_size)
		outputs = self.out(outputs).view(-1, self.word_size)

		return outputs, hidden

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
		output, hidden = decoder(
			x,
			hidden
		)
		outputs.append(output)
		output_onehot = torch.max(torch.softmax(output, dim=1), 1)[1]

		# meet EOS
		if output_onehot.item() == eos_token and not teacher:
			break

		if teacher:
			x = inputs[i+1:i+2]
		else:
			x = output_onehot

	# get (seq, word_size)
	if len(outputs) != 0:
		outputs = torch.cat(outputs, dim=0)
	else:
		outputs = torch.FloatTensor([]).view(0, word_size).to(device)

	return outputs

#compute BLEU-4 score
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
			# print('{:20s} : {}'.format(train_dataset.tenses[i],output_str))
	return blue_score

def KLD_weight_annealing(epoch):
	slope = 0.001
	#slope = 0.1
	scope = (1.0 / slope)*2

	w = (epoch % scope) * slope

	if w > 1.0:
		w = 1.0

	return w

def Teacher_Forcing_Ratio(epoch):
	# from 1.0 to 0.0
	slope = 0.01
	level = 10
	w = 1.0 - (slope * (epoch//level))
	if w <= 0.0:
		w = 0.0

	return w


def KL_loss(m, logvar):
	return torch.sum(0.5 * (-logvar + (m**2) + torch.exp(logvar) - 1))

def trainEpochs( \
	name, encoder, decoder, test_dataset, epoch_size, learning_rate=1e-2, \
	show_size=1000, KLD_weight=0.0, \
	teacher_forcing_ratio = 1.0, eval_size=100, \
	metrics=[],start_epoch=0 \
):
	start = time.time()
	#plots = []
	show_loss_total = 0
	plot_loss_total = 0
	plot_kl_loss_total = 0
	char_accuracy_total = 0
	char_accuracy_len = 0

	kld_w = 0.0
	tfr = 0.0

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

	criterion = nn.CrossEntropyLoss(reduction='sum')

	for epoch in range(start_epoch, epoch_size):
		encoder.train()
		decoder.train()

		if callable(teacher_forcing_ratio):
			tfr = teacher_forcing_ratio(epoch)
		else:
			tfr = teacher_forcing_ratio

		if callable(KLD_weight):
			kld_w = KLD_weight(epoch)
		else:
			kld_w = KLD_weight

		# get data from trian dataset
		for idx in range(len(train_dataset)):
		#for idx in range(1):
			data = train_dataset[idx]
			inputs, c = data

			encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()

			# input no sos and eos
			z, m, logvar = encoder(inputs[1:-1].to(device), encoder.initHidden(), c)

			# decide teacher forcing
			use_teacher_forcing = True if random.random() < tfr else False

			# input has sos
			#outputs, _ = decoder(inputs.to(device), z, encoder.condition(c), use_teacher_forcing)
			outputs = decode_inference(
				decoder, z, encoder.condition(c), maxlen=inputs[1:].size(0),
				teacher=use_teacher_forcing, inputs=inputs.to(device))

			# target no sos
			output_length = outputs.size(0)

			loss = criterion(outputs, inputs[1:1+output_length].to(device))
			kld_loss = KL_loss(m, logvar)
			#loss = criterion(outputs, inputs[:-1].to(device))

			#print('crossentropy : {} , kld : {}'.format(loss.item(), kld_loss.item()))

			(loss + (kld_w * kld_loss)).backward()

			encoder_optimizer.step()
			decoder_optimizer.step()

			show_loss_total += loss.item() + ( kld_w*kld_loss.item() )
			plot_loss_total += loss.item()
			plot_kl_loss_total += kld_loss.item()

			# show output by string
			# outputs_onehot = torch.max(outputs, 1)[1]
			outputs_onehot = torch.max(torch.softmax(outputs, dim=1), 1)[1]
			inputs_str = train_dataset.chardict.stringFromLongtensor(inputs, show_token=True)
			outputs_str = train_dataset.chardict.stringFromLongtensor(outputs_onehot, show_token=True)
			#print(inputs_str,':',outputs_str)

			#char_accuracy_total += (outputs_onehot[:-1] == inputs[1:-1].to(device)).sum().item()
			#char_accuracy_len += len(inputs[1:-1])

			if np.isnan(loss.item()) or np.isnan(kld_loss.item()):
				raise AttributeError("Became NAN !! loss : {}, kl : {}".format(loss.item(), kld_loss.item()))

		score = 0
		for _ in range(eval_size):
			all_score = evaluation(encoder, decoder, test_dataset, show=False)
			score += sum(all_score) / len(all_score)
		score /= eval_size

		# save_model_by_score(
		# 	{'encoder':encoder, 'decoder':decoder},
		# 	score,
		# 	os.path.join('.', name)
		# )

		if (epoch + 1)%show_size == 0:
			show_loss_total /= show_size
			print('epoch: ',epoch)
			all_score = evaluation(encoder, decoder, test_dataset)
			show_loss_total = 0
			torch.save(metrics, 'models/epoch'+str(epoch)+'.pkl')

		metrics.append((plot_loss_total, plot_kl_loss_total, score, \
						kld_w, tfr, learning_rate))

		plot_loss_total = 0
		plot_kl_loss_total = 0
		char_accuracy_total = 0
		char_accuracy_len = 0

	return metrics

def show_curve(df):
	plt.figure(figsize=(10,6))
	plt.title('Training\nLoss/Score/Weight Curve')

	plt.plot(df.index, df.kl, label='KLD', linewidth=3)
	plt.plot(df.index, df.crossentropy, label='CrossEntropy', linewidth=3)

	plt.xlabel('epoch')
	plt.ylabel('loss')

	h1, l1 = plt.gca().get_legend_handles_labels()

	ax = plt.gca().twinx()
	ax.plot(metrics_df.index, metrics_df.score, '.', label='BLEU4-score',c="C2")
	ax.plot(metrics_df.index, metrics_df.klw, '--', label='KLD_weight',c="C3")
	ax.plot(metrics_df.index, metrics_df.tfr, '--', label='Teacher ratio',c="C4")
	ax.set_ylabel('score / weight')

	h2, l2 = ax.get_legend_handles_labels()

	ax.legend(h1+h2, l1+l2)
	plt.show()


def generate_word(encoder, decoder, z, condition, maxlen=20):
	encoder.eval()
	decoder.eval()

	outputs = decode_inference(
		decoder, z, encoder.condition(condition), maxlen=maxlen
	)

	return torch.max(torch.softmax(outputs, dim=1), 1)[1]



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

	encoder = EncoderRNN(word_size, hidden_size, latent_size, num_condition, condition_size).to(device)
	decoder = DecoderRNN(word_size, hidden_size, latent_size, condition_size).to(device)
	metrics = []

	metrics_backup = trainEpochs('training_from_init', encoder, decoder, test_dataset, \
								epoch_size=3, show_size=1, learning_rate=10e-4,\
								KLD_weight=KLD_weight_annealing, \
								teacher_forcing_ratio=Teacher_Forcing_Ratio,\
								metrics=metrics, start_epoch=len(metrics))

	torch.save(metrics, os.path.join('.', 'metrics.pkl'))

	metrics_df = pd.DataFrame(metrics, columns=["crossentropy", "kl", "score", "klw", "tfr", "lr"])
	metrics_df.to_csv(os.path.join('.', 'metrics.csv'), index=False)

	# all_score = evaluation(encoder, decoder, test_dataset)
	# noise = encoder.sample_z()
	# for i in range(len(train_dataset.tenses)):
	# 	outputs = generate_word(encoder, decoder, noise, i)
	# 	output_str = train_dataset.chardict.stringFromLongtensor(outputs)
	# 	print('{:20s} : {}'.format(train_dataset.tenses[i],output_str))

	show_curve(metrics_df)
