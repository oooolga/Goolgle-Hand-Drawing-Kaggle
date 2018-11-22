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

result_path = 'results/'

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_model', required=True, type=str, help='Load model path')
	parser.add_argument('--result_file_name', required=True, type=str, help='Result file name')
	args = parser.parse_args()
	return args

def predict(model, data_loader):
	model.eval()

	prediction_array = np.empty(0, dtype=np.int64)

	for i_batch, batch in tqdm(enumerate(data_loader)):
		data = batch['image'].type(torch.FloatTensor)
		if use_cuda:
			data = data.cuda()

		output = model(data)
		pred = output.data.max(1)[1].cpu().numpy()
		prediction_array = np.concatenate((prediction_array, pred),axis=0)

	pred_matrix = np.stack([np.arange(len(prediction_array)), prediction_array]).T
	return pred_matrix



if __name__ == '__main__':
	# get arguments
	args = parse()

	model = VGGModel(vgg_name='VGG13')
	if use_cuda:
		model.cuda()

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total number of parameters: {}\n'.format(params))

	optimizer = optim.Adam(params=model.parameters(), lr=0, weight_decay=0)

	model, _, _, _, unique_labels = load_checkpoint(args.load_model, model, optimizer)

	test_loader = load_quickdraw_test_data(50)

	pred_matrix = predict(model, test_loader)
	df = get_prediction_dataframe(pred_matrix, unique_labels)
	df.to_csv(os.path.join(result_path, args.result_file_name+'.csv'), sep=',', index=False)

