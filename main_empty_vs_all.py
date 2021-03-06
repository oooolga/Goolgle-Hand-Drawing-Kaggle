import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

use_cuda = torch.cuda.is_available()

import pdb
from tqdm import tqdm
import numpy as np
import argparse, os

from util.load_data import load_quickdraw_data, get_unique_labels, \
						   load_quickdraw_data_empty_vs_all
from util.model_util import save_checkpoint, load_checkpoint
from models.vgg_model import VGGModel
from models.ResNet import resnet
from models.basic_model import BasicModel, crop_images

state = {'train_loss': None,
		 'valid_loss': None,
		 'valid_accuracy': None}

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

		cropped_im = cropped_crop_images(data)

		output = model(data)
		loss = F.cross_entropy(output, target)

		pred = output.data.max(1)[1]
		correct += float(pred.eq(target.data).sum())

		loss_avg += float(loss)

	state['{}_loss'.format(mode)] = loss_avg / len(data_loader)
	state['{}_acc'.format(mode)] = correct / len(data_loader.dataset)



def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
						help='Learning rate.')
	parser.add_argument('--batch_size', default=50, type=int,
						help='Mini-batch size for training.')
	parser.add_argument('--test_batch_size', default=200, type=int,
						help='Mini-batch size for testing.')
	parser.add_argument('--epochs', default=200, type=int,
						help='Total number of epochs.')
	parser.add_argument('--seed', default=123, type=int,
						help='Random number seed.')
	parser.add_argument('--weight_decay', default=5e-8, type=float, help='Weight decay')
	parser.add_argument('--model_name', required=True, type=str, help='Model name')
	parser.add_argument('--load_model', default=None, type=str, help='Load model path')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# get arguments
	args = parse()

	# set seeds
	torch.manual_seed(args.seed)
	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	#model = VGGModel(vgg_name='VGG13')
	#model = resnet(model_name='resnet18', pretrained=False, num_classes=31)
	model_empty_vs_all = BasicModel(nlabels=2)
	if use_cuda:
		model_empty_vs_all.cuda()

	model_parameters = filter(lambda p: p.requires_grad, model_empty_vs_all.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total number of parameters: {}\n'.format(params))

	optimizer = optim.Adam(params=model_empty_vs_all.parameters(), lr=args.learning_rate,
						   weight_decay=args.weight_decay)

	scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

	if args.load_model is None:
		epoch_start = 0
		best_valid_acc = 0
		unique_labels = get_unique_labels()
	else:
		model_empty_vs_all, optimizer, epoch_start, best_valid_acc, unique_labels = \
								load_checkpoint(args.load_model, model_empty_vs_all, optimizer)

	train_empty_vs_all_loader, valid_empty_vs_all_loader = load_quickdraw_data_empty_vs_all(
													batch_size=args.batch_size,
													test_batch_size=args.test_batch_size,
													unique_labels=unique_labels)


	for epoch_i in range(epoch_start, args.epochs+1):
		print('|\tEpoch {}/{}:'.format(epoch_i, args.epochs))
		scheduler.step()

		if epoch_i != 0:
			train(model_empty_vs_all, optimizer, train_empty_vs_all_loader)

		test(model_empty_vs_all, valid_empty_vs_all_loader)

		print('|\t\t[Valid]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['valid_acc'],
																   state['valid_loss']))


		if state['valid_acc'] > best_valid_acc:
			best_valid_acc = state['valid_acc']
			save_checkpoint({
				'epoch_i': epoch_i,
				'state_dict': model_empty_vs_all.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_acc': best_valid_acc,
				'unique_labels': unique_labels
				}, os.path.join(model_path, args.model_name+'.pt'))
