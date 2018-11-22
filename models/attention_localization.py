import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

use_cuda = torch.cuda.is_available()

import pdb

def crop_images(im, kernel_size=30, stride=10, im_size=(100,100)):
	batch_size = im.size(0)
	cropped_im = []

	curr_i, curr_j = 0, 0
	while curr_i+kernel_size <= im_size[0]:
		curr_w = []
		while curr_j+kernel_size <= im_size[1]:
			temp = im[:,:,curr_i:curr_i+kernel_size,
						  curr_j:curr_j+kernel_size]
			yield temp.view(batch_size, -1, kernel_size, kernel_size)
			curr_j += stride
		curr_i += stride
		curr_j = 0

def new_parameter(*size):
	out = nn.Parameter(torch.FloatTensor(*size))
	nn.init.xavier_normal_(out)
	return out

class AttentionLocalizationModel(nn.Module):

	def __init__(self, c_in=1, nlabels=31):
		super(AttentionLocalizationModel, self).__init__()
		self.nlabels = nlabels
		self.c_in = c_in
		# self.layers = 2
		# self.features = nn.Sequential(
		# 	nn.Conv2d(c_in, 32, kernel_size=3, padding=1),
		# 	nn.Dropout2d(p=0.25),
		# 	nn.BatchNorm2d(32),
		# 	nn.ReLU(),
		# 	nn.MaxPool2d(kernel_size=2, stride=2),
		# 	nn.Conv2d(32, 50, kernel_size=3),
		# 	nn.Dropout2d(p=0.25),
		# 	nn.BatchNorm2d(50),
		# 	nn.ReLU(),
		# 	nn.MaxPool2d(kernel_size=2, stride=2),
		# 	nn.Conv2d(50, 50, kernel_size=3),
		# 	nn.Dropout2d(p=0.25),
		# 	nn.BatchNorm2d(50),
		# 	nn.ReLU())#,
		# 	#nn.AdaptiveAvgPool2d((1,1)))
		self.features = nn.Sequential(
			nn.Conv2d(c_in, 64, kernel_size=3),
			#nn.Dropout2d(),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3),
			#nn.Dropout2d(),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=5, stride=3),
			#nn.Dropout2d(),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=3),
			#nn.Dropout2d(),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d((1,1)))

		# self.rnn = nn.GRUCell(256, 256)
		# self.h0 = torch.randn(1, 256)
		# if use_cuda:
		# 	self.h0 = self.h0.cuda()

		self.attention_1 = new_parameter(256, 32)
		self.attention_2 = new_parameter(32, 1)

		self.classifier = nn.Sequential(
			nn.Linear(256, 50),
			nn.ReLU(),
			nn.Linear(50, self.nlabels)
			)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)

			if type(m) in [nn.GRU, nn.LSTM, nn.RNN, nn.GRUCell]:
				for name, param in m.named_parameters():
					if 'weight_ih' in name:
						torch.nn.init.xavier_uniform_(param.data)
					elif 'weight_hh' in name:
						torch.nn.init.orthogonal_(param.data)
					elif 'bias' in name:
						param.data.fill_(0)

	def forward(self, im):
		batch_size = im.size(0)
		cropped_images = crop_images(im)

		# hx = self.h0.repeat(batch_size, 1)

		features = []
		counter = 0
		for next_im in cropped_images:
			# next_im = next(cropped_images)
			feature_i = self.features(next_im).view(batch_size, -1)
			# hx = self.rnn(feature_i, hx)
			features.append(feature_i)
			counter += 1

		features = torch.stack(features).permute(1,0,2)
		attention_score = torch.matmul(F.relu(torch.matmul(features, self.attention_1)), self.attention_2).squeeze()
		# attention_score = torch.matmul(features, self.attention_1).squeeze()
		attention_score = F.softmax(attention_score, dim=1).view(batch_size, counter, 1)
		scored_features = features * attention_score
		condensed_feature = torch.sum(scored_features, dim=1)

		return self.classifier(condensed_feature)
		# return self.classifier(hx)