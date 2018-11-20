import pandas as pd
import numpy as np
import pdb

import torch
from torch.utils.data import Dataset, DataLoader

class QuickDrawDataset(Dataset):

	def __init__(self, train_im_path='./data/train_images.npy',
				 test_im_path='./data/test_images.npy',
				 train_label_path = './data/train_labels.csv',
				 unique_labels = None,
				 mode='train'):

		self.train_im_path = train_im_path
		self.test_im_path = test_im_path
		self.train_label_path = train_label_path
		self.mode = mode

		if mode == 'train' or mode == 'valid':
			train_im = np.load(self.train_im_path, encoding='bytes')
			train_im = np.concatenate(train_im[:,1]).reshape((-1,1,100,100))
			shuffle_idx = np.arange(len(train_im))
			np.random.shuffle(shuffle_idx)
			train_im = train_im[shuffle_idx]

			total_train = int(len(train_im)*0.9)
			train_labels_df = pd.read_csv(self.train_label_path)
			train_labels = self.set_numeric_labels(train_labels_df, unique_labels)
			train_labels = train_labels[shuffle_idx]

			if mode == 'train':
				self.operating_im = train_im[:total_train]
				self.operating_labels = train_labels[:total_train]
			else:
				self.operating_im = train_im[total_train:]
				self.operating_labels = train_labels[total_train:]

		elif mode == 'test':
			test_im = np.load(self.test_im_path, encoding='bytes')
			self.operating_im = np.concatenate(test_im[:,1])\
									.reshape((-1,1,100,100))
			self.operating_labels = None


	def set_numeric_labels(self, labels_df, unique_labels):

		for i in range(len(unique_labels)):
			labels_df.loc[labels_df.Category==unique_labels[i],'Category'] = i
		return labels_df.values[:,1]

	def __len__(self):

		return len(self.operating_im)

	def __getitem__(self, idx):

		if self.mode == 'test':
			return {'image': self.operating_im[idx]}

		return {'image': self.operating_im[idx],
				'label': self.operating_labels[idx]}


class QuickDrawDatasetEmptyVSAll(Dataset):

	def __init__(self, train_im_path='./data/train_images.npy',
				 test_im_path='./data/test_images.npy',
				 train_label_path = './data/train_labels.csv',
				 unique_labels = None,
				 mode='train'):

		self.train_im_path = train_im_path
		self.test_im_path = test_im_path
		self.train_label_path = train_label_path
		self.mode = mode

		if mode == 'train' or mode == 'valid':
			train_im = np.load(self.train_im_path, encoding='bytes')
			train_im = np.concatenate(train_im[:,1]).reshape((-1,1,100,100))
			shuffle_idx = np.arange(len(train_im))
			np.random.shuffle(shuffle_idx)
			train_im = train_im[shuffle_idx]

			train_labels_df = pd.read_csv(self.train_label_path)
			train_labels = self.set_numeric_labels(train_labels_df, unique_labels)
			train_labels = train_labels[shuffle_idx]

			total_data = np.sum(train_labels <= 1)
			total_train = int(0.9*total_data)
			train_idx = np.where(train_labels<=1)[0]

			train_im = train_im[train_idx]
			train_labels = train_labels[train_idx]

			if mode == 'train':
				self.operating_im = train_im[:total_train]
				self.operating_labels = train_labels[:total_train]
			else:
				self.operating_im = train_im[total_train:]
				self.operating_labels = train_labels[total_train:]

	def set_numeric_labels(self, labels_df, unique_labels):
		for i in range(len(unique_labels)):
			if unique_labels[i] == 'empty':
				labels_df.loc[labels_df.Category==unique_labels[i],'Category'] = -1
			else:
				labels_df.loc[labels_df.Category==unique_labels[i],'Category'] = i+1

			values = labels_df.Category.values
			idx = np.where(values==i+1)[0]
			values[idx[:11]] = 0
			labels_df['Category'] = values
		values = labels_df.Category.values
		idx = np.where(values>0)
		values[idx] = 2
		idx = np.where(values<0)
		values[idx] = 1
		labels_df['Category'] = values

		return labels_df.values[:,1]

	def __len__(self):

		return len(self.operating_im)

	def __getitem__(self, idx):

		return {'image': self.operating_im[idx],
				'label': self.operating_labels[idx]}

class QuickDrawDatasetNoEmpty(Dataset):

	def __init__(self, train_im_path='./data/train_images.npy',
				 test_im_path='./data/test_images.npy',
				 train_label_path = './data/train_labels.csv',
				 unique_labels = None,
				 mode='train'):

		self.train_im_path = train_im_path
		self.test_im_path = test_im_path
		self.train_label_path = train_label_path
		self.mode = mode

		if mode == 'train' or mode == 'valid':
			train_im = np.load(self.train_im_path, encoding='bytes')
			train_im = np.concatenate(train_im[:,1]).reshape((-1,1,100,100))

			shuffle_idx = np.arange(len(train_im))
			np.random.shuffle(shuffle_idx)
			train_im = train_im[shuffle_idx]

			train_labels_df = pd.read_csv(self.train_label_path)
			train_labels = self.set_numeric_labels(train_labels_df, unique_labels)
			train_labels = train_labels[shuffle_idx]

			total_data = np.sum(train_labels >= 0)
			total_train = int(0.9*total_data)
			train_idx = np.where(train_labels >= 0)[0]

			train_im = train_im[train_idx]
			train_labels = train_labels[train_idx]

			if mode == 'train':
				self.operating_im = train_im[:total_train]
				self.operating_labels = train_labels[:total_train]
			else:
				self.operating_im = train_im[total_train:]
				self.operating_labels = train_labels[total_train:]

	def set_numeric_labels(self, labels_df, unique_labels):

		tmp = 0
		for i in range(len(unique_labels)):
			if unique_labels[i] == 'empty':
				labels_df.loc[labels_df.Category==unique_labels[i],'Category'] = -1
				tmp = 1
			else:
				labels_df.loc[labels_df.Category==unique_labels[i],'Category'] = i-tmp

		return labels_df.values[:,1]

	def __len__(self):

		return len(self.operating_im)

	def __getitem__(self, idx):

		return {'image': self.operating_im[idx],
				'label': self.operating_labels[idx]}


def get_unique_labels(train_label_path = './data/train_labels.csv'):
	labels_df = pd.read_csv(train_label_path)
	unique_labels = labels_df.Category.unique()
	return unique_labels

def load_quickdraw_test_data(test_batch_size,
							 test_im_path='./data/test_images.npy'):
	test_quickdraw_dataset = QuickDrawDataset(mode='test',
											  test_im_path=test_im_path)
	test_loader = DataLoader(test_quickdraw_dataset, batch_size=test_batch_size)
	return test_loader


def load_quickdraw_data(unique_labels,
						batch_size=50, test_batch_size=200, 
						train_im_path='./data/train_images.npy',
						train_label_path = './data/train_labels.csv'):
		
	train_quickdraw_dataset = QuickDrawDataset(mode='train',
											   unique_labels=unique_labels,
											   train_im_path=train_im_path,
											   train_label_path=train_label_path)
	train_loader = DataLoader(train_quickdraw_dataset, batch_size=batch_size,
							  shuffle=True)

	valid_quickdraw_dataset = QuickDrawDataset(mode='valid',
											   unique_labels=unique_labels,
											   train_im_path=train_im_path,
											   train_label_path=train_label_path)
	valid_loader = DataLoader(valid_quickdraw_dataset, batch_size=test_batch_size)

	return train_loader, valid_loader

def load_quickdraw_data_empty_vs_all(unique_labels,
									 batch_size=50, test_batch_size=200, 
									 train_im_path='./data/train_images.npy',
									 train_label_path = './data/train_labels.csv'):
		
	train_quickdraw_dataset = QuickDrawDatasetEmptyVSAll(mode='train',
											   unique_labels=unique_labels,
											   train_im_path=train_im_path,
											   train_label_path=train_label_path)
	train_loader = DataLoader(train_quickdraw_dataset, batch_size=batch_size,
							  shuffle=True)

	valid_quickdraw_dataset = QuickDrawDatasetEmptyVSAll(mode='valid',
											   unique_labels=unique_labels,
											   train_im_path=train_im_path,
											   train_label_path=train_label_path)
	valid_loader = DataLoader(valid_quickdraw_dataset, batch_size=test_batch_size)

	return train_loader, valid_loader

def load_quickdraw_data_no_empty(unique_labels,
									 batch_size=50, test_batch_size=200, 
									 train_im_path='./data/train_images.npy',
									 train_label_path = './data/train_labels.csv'):
		
	train_quickdraw_dataset = QuickDrawDatasetNoEmpty(mode='train',
											   unique_labels=unique_labels,
											   train_im_path=train_im_path,
											   train_label_path=train_label_path)
	train_loader = DataLoader(train_quickdraw_dataset, batch_size=batch_size,
							  shuffle=True)

	valid_quickdraw_dataset = QuickDrawDatasetNoEmpty(mode='valid',
											   unique_labels=unique_labels,
											   train_im_path=train_im_path,
											   train_label_path=train_label_path)
	valid_loader = DataLoader(valid_quickdraw_dataset, batch_size=test_batch_size)

	return train_loader, valid_loader
