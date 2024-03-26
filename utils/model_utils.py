import torch

def load_model_checkpoint(cfg, optimizer: torch.optim.Optimizer, model_coarse: torch.nn.Module, model_fine: torch.nn.Module):
	"""
	Loads saved models from checkpoint paths.

	Args:
		cfg : Dict like object with user configs
		optimizer (torch.optim.Optimizer): Optimizer to load the state from the checkpoint.
		model_coarse (torch.nn.Module): Coarse model to load the state from the checkpoint.
		model_fine (torch.nn.Module): Fine model to load the state from the checkpoint.

	Returns:
		start_iter (int): Epoch number from which checkpoint was saved.
		torch.optim.Optimizer: Optimizer loaded with the state from the checkpoint.
		torch.nn.Module: Coarse model loaded with the state from the checkpoint.
		torch.nn.Module: Fine model loaded with the state from the checkpoint.
	"""
	try:
		checkpoint = torch.load(cfg.train.checkpoint_path)
	except Exception as e:
		cfg.result.logger.error(f"Unable to load checkpoint: {e}")
	start_iter = checkpoint['epoch']

	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])

	if model_fine != None:
		model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
		return start_iter, optimizer, model_coarse, model_fine
	else:
		return start_iter, optimizer, model_coarse, None