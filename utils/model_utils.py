import torch

def load_checkpoint_model(checkpoint_path:str, optimizer, model_coarse, model_fine):
  '''
  Loading saved models from checkpoint paths

  Args:
  Returns:
  '''
  checkpoint = torch.load(checkpoint_path)
  start_iter = checkpoint['epoch']
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

  # Load model
  model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])
  model_fine.load_state_dict(checkpoint['model_fine_state_dict'])

  return start_iter, optimizer, model_coarse, model_fine