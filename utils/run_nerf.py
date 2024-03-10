import torch
import torch.nn as nn

import numpy as np

from utils.common_utils import get_chunks, sample_pdf
from utils.ray_utils import tf_world2ndc, get_raybundle_for_img, render_image_batch_from_3dinfo, sample_coarse_points
from utils.model_utils import get_radiance_field_per_chunk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO: add comments and func descriptions

def run_nerf(height, width, focal_length, training_campose, use_viewdirs, is_ndc_required, use_white_bkgd,
             near_thresh, far_thresh, num_coarse_samples_per_ray, num_fine_samples_per_ray,
             include_input_in_posenc, include_input_in_direnc, num_pos_encoding_functions,
             num_dir_encoding_functions, model_coarse, model_fine, chunk_size, num_random_rays, mode):
  
	if mode == 'train':
		model_fine.train()
		model_coarse.train()
	else:
		model_fine.eval()
		model_coarse.eval()

	ray_origins, ray_directions = get_raybundle_for_img(height, width, focal_length, training_campose)

	if use_viewdirs: # TODO: check benefit of this
		view_dirs = ray_directions / ray_directions.norm(p=2, dim=-1).unsqueeze(-1)
		view_dirs = view_dirs.view(-1, 3)

	if is_ndc_required:
		ray_origins, ray_directions = tf_world2ndc(ray_origins, ray_directions, near_thresh, height, width, focal_length)

	# Flatten and concatenate
	ray_origins = ray_origins.view(-1, 3)
	ray_directions = ray_directions.view(-1, 3) # h*w, 3

	# Randomly sample rays for OOM error
	if mode=='train' and num_random_rays > 0:
		random_indices = np.random.choice(ray_directions.shape[0], size=(num_random_rays), replace=False)
		ray_directions = ray_directions[random_indices, :] # num_rand_rays x 3
		ray_origins = ray_origins[random_indices, : ] # num_rand_rays x 3
		if use_viewdirs:
			view_dirs = view_dirs[random_indices, : ] 
	
	near_points = near_thresh * torch.ones_like(ray_directions[...,:1])
	far_points = far_thresh * torch.ones_like(ray_directions[...,:1]) # h*w, 1 or num_random_rays, 1
	concatenated_rays = torch.cat((ray_origins, ray_directions, near_points, far_points), dim=-1)
	if use_viewdirs:
		concatenated_rays = torch.cat((concatenated_rays, view_dirs), dim=-1) # h*w, 11 or or num_random_rays, 11

	# Batchify
	chunks = get_chunks(concatenated_rays, chunk_size)
	rgb_map_coarse_list = []
	rgb_map_fine_list = []
	for chunk in chunks:
		ray_origins_batch, ray_directions_batch = chunk[...,:3], chunk[...,3:6]
		near_points_batch, far_points_batch = chunk[...,6], chunk[...,7]
		if use_viewdirs:
			viewdirs_batch = chunk[..., 8:] # h*w, 3

		# Stratified Sampling for generating coarse points
		coarse_depth_values, coarse_sample_points = sample_coarse_points(ray_directions_batch, ray_origins_batch, 
																		near_thresh, far_thresh, num_coarse_samples_per_ray, 
																		is_ndc_required)
		rgba_coarse = get_radiance_field_per_chunk(coarse_sample_points, model_coarse, num_pos_encoding_functions, 
												include_input_in_posenc, use_viewdirs, viewdirs_batch,
												num_dir_encoding_functions, include_input_in_direnc, chunk_size)
		# TODO: 1. Check extra arguments - white_bckgd, noise
		# TODO: 2. shud we mutiply by ray_directions ?
		rgb_map_coarse, disp_map_coarse, acc_map_coarse, depth_map_coarse, weights = render_image_batch_from_3dinfo(rgba_coarse , coarse_depth_values, use_white_bkgd) 
		rgb_map_coarse_list.append(rgb_map_coarse)

		# Fine - Hierachical Sampling
		if num_fine_samples_per_ray > 0:
			# TODO: Check everything from here ??????
			depth_values_mid = .5 * (coarse_depth_values[..., 1:] + coarse_depth_values[..., :-1])
			z_samples = sample_pdf(depth_values_mid, weights[..., 1:-1], num_fine_samples_per_ray, True)
			z_samples = z_samples.detach()

			fine_depth_values, _ = torch.sort(torch.cat((coarse_depth_values, z_samples), dim=-1), dim=-1)
			fine_sample_points = ray_origins_batch[..., None, :] + ray_directions_batch[..., None, :] * fine_depth_values[..., :, None]
		
			rgba_fine = get_radiance_field_per_chunk(fine_sample_points, model_fine, num_pos_encoding_functions, 
												include_input_in_posenc, use_viewdirs, viewdirs_batch,
												num_dir_encoding_functions, include_input_in_direnc, chunk_size)

			rgb_map_fine, disp_map_fine, acc_map_fine, depth_map_fine, _ = render_image_batch_from_3dinfo(rgba_fine , fine_depth_values, use_white_bkgd) 
			rgb_map_fine_list.append(rgb_map_fine)

	rgb_coarse_image = torch.cat(rgb_map_coarse_list, dim=0)
	rgb_fine_image = torch.cat(rgb_map_fine_list, dim=0)

	return rgb_coarse_image, rgb_fine_image, random_indices