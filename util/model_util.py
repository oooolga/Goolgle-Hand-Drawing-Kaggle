import torch, os
import torch.optim as optim

def save_checkpoint(state, save_path):
	torch.save(state, save_path)
	print('Finished saving model: {}'.format(save_path))


def load_checkpoint(model_path, model, optimizer):
	if model_path and os.path.isfile(model_path):
		checkpoint = torch.load(model_path)
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch_i = checkpoint['epoch_i']
		best_valid_loss = checkpoint['best_loss']
		unique_labels = checkpoint['unique_labels']
	else:
		print('File {} not found.'.format(model_name))
		raise FileNotFoundError

	return model, optimizer, epoch_i, best_valid_loss, unique_labels
