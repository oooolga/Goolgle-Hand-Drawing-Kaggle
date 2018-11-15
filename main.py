import torch
import pdb

from .util.load_data import QuickDrawDataset

if __name__ == '__main__':
	data = QuickDrawDataset()

	l = len(data)

	pdb.set_trace()