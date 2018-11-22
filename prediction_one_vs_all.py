import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

use_cuda = torch.cuda.is_available()

import pdb
from tqdm import tqdm
import numpy as np
import argparse, os

from util.load_data import load_quickdraw_data, get_unique_labels, load_quickdraw_test_data
from util.save_prediction import get_prediction_dataframe
from util.model_util import save_checkpoint, load_checkpoint
from models.vgg_model import VGGModel
from models.basic_model import BasicModel
from models.attention_localization import AttentionLocalizationModel

result_path = 'results/'

state = {'train_accuracy': None,
		 'valid_accuracy': None}

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_model', required=True, type=str, help='Load model path')
	parser.add_argument('--load_empty_model', required=True, type=str, help='Load empty model path')
	parser.add_argument('--result_file_name', required=True, type=str, help='Result file name')
	args = parser.parse_args()
	return args

def predict(model_empty_vs_all, model, data_loader):
	model_empty_vs_all.eval()
	model.eval()

	prediction_array = np.empty(0, dtype=np.int64)

	with torch.no_grad():
		for i_batch, batch in tqdm(enumerate(data_loader)):
			data = batch['image'].type(torch.FloatTensor)
			if use_cuda:
				data = data.cuda()

			output = model(data)
			zero_output = model_empty_vs_all(data)

			class_prob, pred = output.data.max(1)
			zero_prob, zero_pred = zero_output.data.max(1)

			pdb.set_trace()

			zero_idx = np.where(zero_pred==1)
			zeros_pred = np.where(zero_prob[zero_idx]>3)
			zero_idx = zero_idx[0][zeros_pred[0]]
			zero_idx_p = np.where(class_prob<1.26)[0]
			pred[np.intersect1d(zero_idx, zero_idx_p)] = 21
			pred[pred>=21] += 1
			prediction_array = np.concatenate((prediction_array, pred),axis=0)

	pred_matrix = np.stack([np.arange(len(prediction_array)), prediction_array]).T
	return pred_matrix

def evaluate(model_empty_vs_all, model, data_loader, unique_labels, mode='train'):
	model_empty_vs_all.eval()
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
			zero_output = model_empty_vs_all(data)

			class_prob, pred = output.data.max(1)
			zero_prob, zero_pred = zero_output.data.max(1)
			pdb.set_trace()
			
			zero_idx = np.where(zero_pred==1)
			zeros_pred = np.where(zero_prob[zero_idx]>3)
			zero_idx = zero_idx[0][zeros_pred[0]]
			zero_idx_p = np.where(class_prob<1.26)[0]
			pred[np.intersect1d(zero_idx, zero_idx_p)] = 21
			pred[pred>=21] += 1

			pdb.set_trace()

			correct += float(pred.eq(target.data).sum())

			del output, data, target

	state['{}_acc'.format(mode)] = correct / len(data_loader.dataset)

if __name__ == '__main__':
	# get arguments
	args = parse()

	model_empty_vs_all = BasicModel(nlabels=2)
	if use_cuda:
		model_empty_vs_all.cuda()
	model_parameters = filter(lambda p: p.requires_grad, model_empty_vs_all.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total number of parameters: {}\n'.format(params))


	model = AttentionLocalizationModel(nlabels=30)
	if use_cuda:
		model.cuda()

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total number of parameters: {}\n'.format(params))

	optimizer_empty_vs_all = optim.Adam(params=model_empty_vs_all.parameters(), lr=0, weight_decay=0)
	optimizer = optim.Adam(params=model.parameters(), lr=0, weight_decay=0)

	model_empty_vs_all, _, _, _, _ = load_checkpoint(args.load_empty_model, model_empty_vs_all,
													 optimizer_empty_vs_all)
	model, _, _, _, unique_labels = load_checkpoint(args.load_model, model, optimizer)

	train_loader, valid_loader = load_quickdraw_data(
													batch_size=200,
													test_batch_size=200,
													unique_labels=unique_labels)

	evaluate(model_empty_vs_all, model, train_loader, unique_labels)
	evaluate(model_empty_vs_all, model, valid_loader, unique_labels, mode='valid')

	print(state['train_acc'])
	print(state['valid_acc'])

	# test_loader = load_quickdraw_test_data(50)

	# pred_matrix = predict(model_empty_vs_all, model, test_loader)
	# # pdb.set_trace()
	# df = get_prediction_dataframe(pred_matrix, unique_labels)
	# df.to_csv(os.path.join(result_path, args.result_file_name+'.csv'), sep=',', index=False)

