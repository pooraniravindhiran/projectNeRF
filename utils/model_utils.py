import torch

from utils.common_utils import positional_encoding
from utils.common_utils import get_chunks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_checkpoint(checkpoint_path, optimizer, model_coarse, model_fine):
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

def get_radiance_field_per_chunk(sample_points, model, num_pos_encoding_functions, include_input_in_posenc, use_viewdirs,
								 viewdirs_batch, num_dir_encoding_functions, include_input_in_direnc, chunk_size):
	# Flatten sample points
	flattened_sample_points = sample_points.view(-1,3) # h*w*num, 3

	# Encode sample points using positional embedding
	encoded_sample_points = positional_encoding(flattened_sample_points, num_pos_encoding_functions, include_input_in_posenc)
	if use_viewdirs:
		ipdirs_batch = viewdirs_batch[...,None,:].expand(sample_points.shape) # h*w,num,3
		ipdirs_batch = ipdirs_batch.reshape(-1, 3) # h*w*num, 3
		encoded_dirs = positional_encoding(ipdirs_batch, num_dir_encoding_functions, include_input_in_direnc)
		encoded_sample_points_batch = torch.cat((encoded_sample_points, encoded_dirs), dim=-1)
	
	# Batchify
	chunks = get_chunks(encoded_sample_points_batch, chunk_size)
	rgba_batch = []
	for chunk in chunks:
		rgba_batch.append(model(chunk.to(device)))
	rgba_batch = torch.cat(rgba_batch, dim=0)
	rgba_batch = rgba_batch.reshape(list(sample_points.shape[:-1]) + [rgba_batch.shape[-1]])
	return rgba_batch


def get_radiance_field_per_chunk_mip(mu_tensor, diag_sigma_tensor, model, num_pos_encoding_functions, include_input_in_posenc, use_viewdirs,
								 viewdirs_batch, num_dir_encoding_functions, include_input_in_direnc, chunk_size):

		# Encode sample points using integrated positional embedding
		encoded_sample_points = integrated_positional_encoding(mu_tensor, diag_sigma_tensor, num_pos_encoding_functions)
		# encoded_sample_points = positional_encoding_obj(mu_tensor, diag_sigma_tensor)[0]
		encoded_sample_points = encoded_sample_points.reshape(-1, encoded_sample_points.shape[-1])
		# print(encoded_sample_points.shape)

		if use_viewdirs:
				ipdirs_batch = viewdirs_batch[...,None,:].expand(mu_tensor.shape) # h*w,num,3
				ipdirs_batch = ipdirs_batch.reshape(-1, 3) # h*w*num, 3
				# encoded_dirs = viewdirs_encoding_obj(ipdirs_batch.to(device))
				encoded_dirs = positional_encoding(ipdirs_batch, num_dir_encoding_functions, include_input_in_direnc)
				# print(encoded_dirs.shape)
				encoded_sample_points = torch.cat((encoded_sample_points, encoded_dirs), dim=-1)

		# print(encoded_sample_points.shape)
		# Batchify
		ip_chunks = get_chunks(encoded_sample_points, chunk_size)
		rgba_batch = []
		for chunk in ip_chunks:
			rgba_batch.append(model(chunk.to(device)))

		rgba_batch = torch.cat(rgba_batch, dim=0)
		rgba_batch = rgba_batch.reshape(list(mu_tensor.shape[:-1]) + [rgba_batch.shape[-1]])
		return rgba_batch