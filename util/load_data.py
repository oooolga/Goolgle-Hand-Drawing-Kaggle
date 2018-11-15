import panda as pd
import numpy as np
import pdb

import torch
from torch.utils.data import Dataset, DataLoader

class QuickDrawDataset(Dataset):

	def __init__(self, train_im_path='./data/train_images.npy',
				 test_im_path='./data/test_images.npy',
				 train_label_path = './data/train_labels.csv'):

		self.train_im_path = train_im_path
		self.test_im_path = test_im_path
		self.train_label_path = train_label_path

		self.train_im = np.load(self.train_im_path)
		self.test_im = np.load(self.test_im_path)
		self.train_label = pd.read_csv(self.train_label_path)
		self.mode = 'train'

	def set_train_mode(self):
		self.mode = 'train'

	def set_validation_mode(self):
		self.mode = 'valid'

	def set_test_mode(self):
		self.mode = 'test'

	def __len__(self):

		pdb.set_trace()