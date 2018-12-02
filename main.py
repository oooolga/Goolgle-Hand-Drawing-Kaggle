import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

use_cuda = torch.cuda.is_available()

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pdb
from tqdm import tqdm
import numpy as np
import argparse, os

from util.model_util import save_checkpoint, load_checkpoint
from util.load_data import load_quickdraw_data, get_unique_labels, load_quickdraw_data_all
from models.vgg_model import VGGModel
from models.ResNet import resnet
from models.basic_model import BasicModel
from models.attention_localization import AttentionLocalizationModel

state = {'train_loss': [],
		 'train_acc': [],
		 'valid_loss': [],
		 'valid_acc': []}
plot_state = {'train_loss': [],
			  'train_acc': [],
			  'valid_loss': [],
			  'valid_acc': [],
			  'epochs': []}

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

	with torch.no_grad():
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

			del output, data, target

	state['{}_loss'.format(mode)].append(loss_avg / len(data_loader))
	state['{}_acc'.format(mode)].append(correct / len(data_loader.dataset))



def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=5e-2, type=float,
						help='Learning rate.')
	parser.add_argument('--batch_size', default=50, type=int,
						help='Mini-batch size for training.')
	parser.add_argument('--test_batch_size', default=200, type=int,
						help='Mini-batch size for testing.')
	parser.add_argument('--epochs', default=200, type=int,
						help='Total number of epochs.')
	parser.add_argument('--seed', default=123, type=int,
						help='Random number seed.')
	parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay.')
	parser.add_argument('--model_name', required=True, type=str, help='Model name.')
	parser.add_argument('--load_model', default=None, type=str, help='Load model path.')
	parser.add_argument('--optimizer', default='Adam', type=str, help='Optimizer type.')
	parser.add_argument('--load_all_train', action='store_true', help='Load all data as train flag.')
	parser.add_argument('--plot_path', default='results', type=str, help='Path for plots.')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# get arguments
	args = parse()

	# set seeds
	torch.manual_seed(args.seed)
	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	model = VGGModel(vgg_name='VGG13')
	# model = resnet(model_name='resnet18', pretrained=False, num_classes=31)
	# model = AttentionLocalizationModel(nlabels=31)
	#model = BasicModel()
	if use_cuda:
		model.cuda()

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total number of parameters: {}\n'.format(params))

	if args.optimizer == 'Adam':

		optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate,
							   weight_decay=args.weight_decay)
	else:
		optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.9,
							  weight_decay=args.weight_decay)

	scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

	if args.load_model is None:
		epoch_start = 0
		best_valid_acc = 0
		unique_labels = get_unique_labels()
	else:
		model, optimizer, epoch_start, best_valid_acc, unique_labels = \
								load_checkpoint(args.load_model, model, optimizer)

	if args.load_all_train:
		train_loader = load_quickdraw_data_all(batch_size=args.batch_size,
											   unique_labels=unique_labels)
	else:
		train_loader, valid_loader = load_quickdraw_data(
														batch_size=args.batch_size,
														test_batch_size=args.test_batch_size,
														unique_labels=unique_labels)

	plot_freq = 5

	for epoch_i in range(epoch_start, args.epochs+1):
		print('|\tEpoch {}/{}:'.format(epoch_i, args.epochs))
		scheduler.step()

		if epoch_i != 0:
			train(model, optimizer, train_loader)

		if not args.load_all_train:
			test(model, valid_loader)
		test(model, train_loader, mode='train')

		print('|\t\t[Train]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['train_acc'][-1],
																   state['train_loss'][-1]))

		if not args.load_all_train:
			print('|\t\t[Valid]:\taccuracy={:.3f}\tloss={:.3f}'.format(state['valid_acc'][-1],
																	   state['valid_loss'][-1]))
		else:
			state['valid_acc'].append(state['train_acc'][-1])

		if epoch_i%plot_freq == 0:
			for k in state:
				plot_state[k].append(state[k][-1])
			plot_state['epochs'].append(epoch_i)

		if state['valid_acc'][-1] > best_valid_acc:
			best_valid_acc = state['valid_acc'][-1]
			save_checkpoint({
				'epoch_i': epoch_i,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_acc': best_valid_acc,
				'unique_labels': unique_labels
				}, os.path.join(model_path, args.model_name+'.pt'))

	plt.close('all')
	fig, ax1 = plt.subplots()
	line_ta = ax1.plot(plot_state['epochs'], plot_state['train_acc'], color="#7aa0c4", label='train acc')
	line_va = ax1.plot(plot_state['epochs'], plot_state['valid_acc'], color="#ca82e1", label='valid acc')
	ax1.set_xlabel('epoch')
	ax1.set_ylabel('accuracy')

	ax2 = ax1.twinx()
	line_tl = ax2.plot(plot_state['epochs'], plot_state['train_loss'], color="#8bcd50", label='train loss')
	line_vl = ax2.plot(plot_state['epochs'], plot_state['valid_loss'], color="#e18882", label='valid loss')
	ax2.set_ylabel('loss')

	lines = line_ta + line_va + line_tl + line_vl
	labs = [l.get_label() for l in lines]

	#fig.subplots_adjust(right=0.75) 
	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0 + box.height * 0.1,
					box.width, box.height * 0.9])
	ax1.legend(lines, labs, loc='upper center', bbox_to_anchor=(0.5, -0.05),
     				 fancybox=True, shadow=True, ncol=5)
	fig.tight_layout()
	plt.title('Training curves')
	plt.savefig(os.path.join(args.plot_path, args.model_name+'_training_curve.png'),
				bbox_inches='tight')
	plt.clf()
	plt.close('all')
