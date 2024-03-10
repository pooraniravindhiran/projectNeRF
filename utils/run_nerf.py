import torch
import torch.nn as nn
import numpy as np
from utils.common_utils import get_chunks
from utils.ray_utils import tf_world2ndc, get_raybundle_for_img, render_image_batch_from_3dinfo, sample_coarse_points
from utils.common_utils import positional_encoding, sample_pdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_Nerf(height, width, focal_length, training_campose, use_viewdirs, is_ndc_required, use_white_bkgd,
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

  # Random Rays Sampling
  if num_random_rays > 0:
    random_indices = np.random.choice(ray_directions.shape[0], size=(num_random_rays), replace=False)
    ray_directions = ray_directions[random_indices, :] # num_rand_rays x 3
    ray_origins = ray_origins[random_indices, : ] # num_rand_rays x 3


  near_points = near_thresh * torch.ones_like(ray_directions[...,:1])
  far_points = far_thresh * torch.ones_like(ray_directions[...,:1]) # h*w, 1 or num_random_rays, 1
  # print(ray_origins.shape, ray_directions.shape, near_points.shape, far_points.shape)
  concatenated_rays = torch.cat((ray_origins, ray_directions, near_points, far_points), dim=-1)
  if use_viewdirs:
    concatenated_rays = torch.cat((concatenated_rays, view_dirs), dim=-1) # h*w, 11 or or num_random_rays, 11
  # print(concatenated_rays.shape)

  # Batchify
  chunks = get_chunks(concatenated_rays, chunk_size)
  rgb_map_coarse_list = []
  # acc_map_coarse_list = []
  # disp_map_coarse_list = []

  rgb_map_fine_list = []
  # acc_map_fine_list = []
  # disp_map_fine_list = []

  for chunk in chunks:
    ray_origins_batch, ray_directions_batch = chunk[...,:3], chunk[...,3:6]
    near_points_batch, far_points_batch = chunk[...,6], chunk[...,7]
    if use_viewdirs:
      viewdirs_batch = chunk[..., 8:] # h*w, 3

    # Stratified Sampling for generating coarse points
    coarse_depth_values, coarse_sample_points = sample_coarse_points(ray_directions_batch, ray_origins_batch, near_thresh, far_thresh, num_coarse_samples_per_ray, is_ndc_required)
    # Flatten sample points
    flattened_coarse_sample_points = coarse_sample_points.view(-1,3) # h*w*num, 3

    # Encode sample points using positional embedding
    encoded_coarse_sample_points = positional_encoding(flattened_coarse_sample_points, num_pos_encoding_functions, include_input_in_posenc)
    if use_viewdirs:
      ipdirs_batch = viewdirs_batch[...,None,:].expand(coarse_sample_points.shape) # h*w,num,3
      # print(ipdirs_batch.shape)
      ipdirs_batch = ipdirs_batch.reshape(-1, 3) # h*w*num, 3
      # print(ipdirs_batch.shape)
      encoded_dirs = positional_encoding(ipdirs_batch, num_dir_encoding_functions, include_input_in_direnc)
      # print(encoded_coarse_sample_points.shape, encoded_dirs.shape)
      encoded_coarse_sample_points_batch = torch.cat((encoded_coarse_sample_points, encoded_dirs), dim=-1)
      # print(encoded_coarse_sample_points.shape)

    rgba_coarse = model_coarse(encoded_coarse_sample_points_batch.to(device))

    rgba_coarse = rgba_coarse.reshape(list(coarse_sample_points.shape[:-1]) + [rgba_coarse.shape[-1]])
    # print(rgba_coarse.shape)
    # TODO: 1. Check extra arguments - white_bckgd, noise
    # TODO: 2. shud we mutiply by ray_directions ?
    rgb_map_coarse, disp_map_coarse, acc_map_coarse, depth_map_coarse, weights = render_image_batch_from_3dinfo(rgba_coarse , coarse_depth_values, use_white_bkgd) 
    # print(rgb_map_coarse.shape, weights.shape)
    rgb_map_coarse_list.append(rgb_map_coarse)
    # acc_map_coarse_list.append(acc_map_coarse)
    # disp_map_coarse_list.append(disp_map_coarse)

    # Fine - Hierachical Sampling
    if num_fine_samples_per_ray > 0:
      # TODO: Check everything from here ??????
      depth_values_mid = .5 * (coarse_depth_values[..., 1:] + coarse_depth_values[..., :-1])
      z_samples = sample_pdf(depth_values_mid, weights[..., 1:-1], num_fine_samples_per_ray, True)
      z_samples = z_samples.detach()

      fine_depth_values, _ = torch.sort(torch.cat((coarse_depth_values, z_samples), dim=-1), dim=-1)
      fine_sample_points = ray_origins_batch[..., None, :] + ray_directions_batch[..., None, :] * fine_depth_values[..., :, None]
      # print(fine_sample_points.shape)
      
      # Flatten fine sample points
      flattened_fine_sample_points = fine_sample_points.view(-1,3) # h*w*num, 3 

      # Encode fine sample points using positional embedding
      encoded_fine_sample_points = positional_encoding(flattened_fine_sample_points, num_pos_encoding_functions, include_input_in_posenc)
      # print(encoded_fine_sample_points.shape)
      if use_viewdirs:
        ipdirs_batch = viewdirs_batch[...,None,:].expand(fine_sample_points.shape) # h*w,num,3
        # print(ipdirs_batch.shape)
        ipdirs_batch = ipdirs_batch.reshape(-1, 3) # h*w*num, 3
        # print(ipdirs_batch.shape)
        encoded_dirs = positional_encoding(ipdirs_batch, num_dir_encoding_functions, include_input_in_direnc)
        # print(encoded_coarse_sample_points.shape, encoded_dirs.shape)
        encoded_fine_sample_points_batch = torch.cat((encoded_fine_sample_points, encoded_dirs), dim=-1)
        # print(encoded_coarse_sample_points.shape)

      # print(encoded_fine_sample_points_batch.shape)

      rgba_fine = model_fine(encoded_fine_sample_points_batch.to(device))

      rgba_fine = rgba_fine.reshape(list(fine_sample_points.shape[:-1]) + [rgba_fine.shape[-1]])


      # TODO: 1. Check extra arguments - white_bckgd, noise
      # TODO: 2. shud we mutiply by ray_directions ?
      rgb_map_fine, disp_map_fine, acc_map_fine, depth_map_fine, _ = render_image_batch_from_3dinfo(rgba_coarse , coarse_depth_values, use_white_bkgd) 
      # print(rgb_map_coarse.shape, weights.shape)
      rgb_map_fine_list.append(rgb_map_fine)
      # acc_map_fine_list.append(acc_map_fine)
      # disp_map_fine_list.append(disp_map_fine)

  rgb_coarse_image = torch.cat(rgb_map_coarse_list, dim=0)
  rgb_fine_image = torch.cat(rgb_map_fine_list, dim=0)

  return rgb_coarse_image, rgb_fine_image
      
        
