import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import pdb

cfg = {
	'VGG8': [32, 'M', 32, 'M', 32, 'M', 64, 'M', 64, 'M'],
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGGModel(nn.Module):

	def __init__(self, c_in=1, nlabels=31, vgg_name='VGG11'):
		super(VGGModel, self).__init__()
		self.nlabels = nlabels
		self.c_in = c_in
		self.features = self._make_layers(cfg[vgg_name])
		self.classifier = nn.Linear(cfg[vgg_name][-2], self.nlabels)

		nn.init.kaiming_normal_(self.classifier.weight)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg):
		layers = []
		in_channels = self.c_in
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AdaptiveAvgPool2d((1,1))]
		return nn.Sequential(*layers)