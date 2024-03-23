import torch

def load_model_checkpoint(checkpoint_path: str, optimizer: torch.optim.Optimizer, model_coarse: torch.nn.Module, model_fine: torch.nn.Module):
	"""
	Loads saved models from checkpoint paths.

	Args:
		checkpoint_path (str): Path to the checkpoint file.
		optimizer (torch.optim.Optimizer): Optimizer to load the state from the checkpoint.
		model_coarse (torch.nn.Module): Coarse model to load the state from the checkpoint.
		model_fine (torch.nn.Module): Fine model to load the state from the checkpoint.

	Returns:
		start_iter (int): Epoch number from which checkpoint was saved.
		torch.optim.Optimizer: Optimizer loaded with the state from the checkpoint.
		torch.nn.Module: Coarse model loaded with the state from the checkpoint.
		torch.nn.Module: Fine model loaded with the state from the checkpoint.
	"""
	checkpoint = torch.load(checkpoint_path)
	start_iter = checkpoint['epoch']

	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])
	model_fine.load_state_dict(checkpoint['model_fine_state_dict'])

	return start_iter, optimizer, model_coarse, model_fine