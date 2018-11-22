import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pdb

class BasicModel(nn.Module):
	def __init__(self, c_in=1, nlabels=31):
		super(BasicModel, self).__init__()
		self.nlabels = nlabels
		self.c_in = c_in
		self.features = nn.Sequential(
			nn.Conv2d(c_in, 64, kernel_size=3, padding=1),
			nn.Dropout2d(),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 64, kernel_size=5),
			nn.Dropout2d(),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 128, kernel_size=5, stride=3),
			nn.Dropout2d(),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(128, 128, kernel_size=3),
			nn.Dropout2d(),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True))
		self.classifier = nn.Linear(128, nlabels)

		nn.init.kaiming_normal_(self.classifier.weight)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		batch_size = x.size(0)
		out = self.features(x).view(batch_size, -1)
		out = self.classifier(out)
		return out


def crop_images(im, kernel_size=30, stride=2, im_size=(100,100)):
	batch_size = im.size(0)
	cropped_im = []

	curr_i, curr_j = 0, 0
	while curr_i+kernel_size <= im_size[0]:
		curr_w = []
		while curr_j+kernel_size <= im_size[1]:
			curr_w.append(im[:,0,curr_i:curr_i+kernel_size,
						  curr_j:curr_j+kernel_size])
			curr_j += stride
		cropped_im.append(torch.stack(curr_w))
		curr_i += stride
		curr_j = 0

	cropped_im = torch.stack(cropped_im).permute(2,0,1,3,4).view(batch_size,-1,
																 kernel_size,
																 kernel_size)
	return cropped_im