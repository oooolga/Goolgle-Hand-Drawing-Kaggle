import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

use_cuda = torch.cuda.is_available()

import pdb
from tqdm import tqdm
import numpy as np
import argparse, os

from util.load_data import load_quickdraw_data
from util.model_util import save_checkpoint
from models.vgg_model import Model

state = {'train_loss': None,
		 'valid_loss': None,
		 'valid_accuracy': None}

best_valid_loss = float('inf')

model_path = 'saved_models/'

def train(model, optimizer, train_loader):
	model.train()

	for i_batch, batch in tqdm(enumerate(train_loader)):

		data, target = batch['image'].type(torch.FloatTensor), \
					   batch['label'].type(torch.long)

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		output = model(data)
		optimizer.zero_grad()
		loss = F.cross_entropy(output, target)
		loss.backward()
		optimizer.step()

def test(model, data_loader, mode='valid'):
	model.eval()

	loss_avg = 0.0
	correct = 0

	for i_batch, batch in tqdm(enumerate(data_loader)):

		data, target = batch['image'].type(torch.FloatTensor), \
					   batch['label'].type(torch.long)

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		output = model(data)
		loss = F.cross_entropy(output, target)

		pred = output.data.max(1)[1]
		correct += float(pred.eq(target.data).sum())

		loss_avg += float(loss)

	state['{}_loss'.format(mode)] = loss_avg / len(data_loader)
	state['{}_acc'.format(mode)] = correct / len(data_loader.dataset)


def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=5e-3, type=float,
						help='Learning rate.')
	parser.add_argument('--batch_size', default=50, type=int,
						help='Mini-batch size for training.')
	parser.add_argument('--test_batch_size', default=200, type=int,
						help='Mini-batch size for testing.')
	parser.add_argument('--epochs', default=200, type=int,
						help='Total number of epochs.')
	parser.add_argument('--seed', default=123, type=int,
						help='Random number seed.')
	parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
	parser.add_argument('--model_name', default='VGG', type=str, help='Model name')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# get arguments
	args = parse()

	# set seeds
	torch.manual_seed(args.seed)
	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	train_loader, valid_loader, test_loader, unique_labels = load_quickdraw_data(batch_size=args.batch_size,
																test_batch_size=args.test_batch_size)

	model = Model()
	if use_cuda:
		model.cuda()

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total number of parameters: {}\n'.format(params))

	optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate,
						   eps=1e-07,
						   weight_decay=args.weight_decay)

	scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

	epoch_start = 0

	for epoch_i in range(epoch_start, args.epochs+1):
		print('|\tEpoch {}/{}:'.format(epoch_i, args.epochs))
		scheduler.step()

		if epoch_i != 0:
			train(model, optimizer, train_loader)

		test(model, valid_loader)

		print('|\t\t[Valid]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['valid_acc'],
																   state['valid_loss']))


		if state['valid_loss'] < best_valid_loss:
			best_valid_loss = state['valid_loss']
			save_checkpoint({
				'epoch_i': epoch_i,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_loss': best_valid_loss,
				'unique_labels': unique_labels
				}, os.path.join(model_path, args.model_name+'.pt'))
