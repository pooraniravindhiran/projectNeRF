import torch

import numpy as np

from utils.common_utils import get_chunks, sample_pdf
from utils.mip_ray_utils import *
from utils.ray_utils import get_raybundle_for_img, tf_world2ndc

def run_mipnerf(height: int, width: int, focal_length: float, pose: torch.Tensor, 
			 near_thresh: float, far_thresh: float, model: torch.nn.Module, 
			 cfg, epoch_num: int, mode: str):
	"""
    Run NeRF (Neural Radiance Fields) to render images for given poses.

    Args:
        height (int): Height of the image.
        width (int): Width of the image.
        focal_length (float): Focal length of the camera.
        pose (torch.Tensor): Pose of the camera.
        near_thresh (float): Near threshold for depth.
        far_thresh (float): Far threshold for depth.
        model (torch.nn.Module): Model for NeRF.
        cfg (object): Dictionary like configuration object.
		epoch_num (int): Iteration number
        mode (str): Mode of operation ('train' or 'eval').
		
    Returns:
        rgb_coarse_image (torch.Tensor): Predicted coarse RGB image.
        rgb_fine_image (torch.Tensor): Predicted fine RGB image.
        selected_ray_indices (numpy.ndarray): Selected ray indices that are used for the prediction.
    """
  
    # Check if the model is training or on inference
	if mode == 'train':
		model.train()
	else:
		model.eval()

    # Get rays passing from projection center through every pixel of the image
	ray_origins, ray_directions = get_raybundle_for_img(height, width, focal_length, pose, cfg.device)
	if cfg.model.use_viewdirs: # TODO: check benefit of this
		view_dirs = ray_directions / ray_directions.norm(p=2, dim=-1).unsqueeze(-1)

    # Convert to Normalized Device Coordinate space for forward facing scenes
	if cfg.dataset.is_ndc_required:
		ray_origins, ray_directions = tf_world2ndc(ray_origins, ray_directions, near_thresh, height, width, focal_length)
		
    # Euclidean distance between each direction vector and its right neigbor along x axis except last ray, left neighbor
	dx = torch.sqrt(torch.sum(((ray_directions[:,:-1,:] - ray_directions[:,1:,:]) ** 2), dim=2))
	dx = torch.concat((dx, dx[:, -2:-1]), dim=-1)
	cone_radii = dx * 2/ np.sqrt(12) # (h, w)

	# Randomly sample rays and use only selected rays for 3D rendering to avoid OOM error
	# ray params will then have dimension (num_selected_rays, 3)
	selected_ray_coors = None
	if mode=='train' and cfg.model.num_selected_rays > 0:
		if epoch_num < cfg.model.centercrop_epochs:
			dheight = int(height//2 * 0.5)
			dwidth = int(width//2 * 0.5)
			img_coors = torch.stack(torch.meshgrid(
				torch.linspace(height//2-dheight, height//2+dheight-1, 2*dheight),
				torch.linspace(width//2-dwidth, width//2+dwidth-1, 2*dwidth)), -1)
		else:
			img_coors = torch.stack(torch.meshgrid(
				torch.linspace(0, height-1, height),
				torch.linspace(0, width-1, width)), -1)
		img_coors = img_coors.reshape(-1, 2)
		selected_ray_indices = np.random.choice(img_coors.shape[0], size=(cfg.model.num_selected_rays), replace=False)
		selected_ray_coors = img_coors[selected_ray_indices].long()
		ray_origins = ray_origins[selected_ray_coors[:, 0], selected_ray_coors[:, 1]]
		ray_directions = ray_directions[selected_ray_coors[:, 0], selected_ray_coors[:, 1]]
		cone_radii = cone_radii[selected_ray_coors[:, 0], selected_ray_coors[:, 1]].unsqueeze(-1)
		if cfg.model.use_viewdirs:
			view_dirs = view_dirs[selected_ray_coors[:, 0], selected_ray_coors[:, 1]]
	else:
		ray_origins = ray_origins.view(-1, 3)
		ray_directions = ray_directions.view(-1, 3) # h*w, 3
		view_dirs = view_dirs.view(-1, 3)
		cone_radii = cone_radii.view(-1, 1)
	
    # Concatenate all necessary fields required for 3D rendering
	# Concatednated result is of dimension (h*w, 10) or (num_selected_rays, 10)
	concatenated_rays = torch.cat((ray_origins, ray_directions, cone_radii), dim=-1)
	if cfg.model.use_viewdirs:
		concatenated_rays = torch.cat((concatenated_rays, view_dirs), dim=-1)

	# Similar to making image batches in CNNs, here we batch the rays together for processing and iterate through them
	chunks = get_chunks(concatenated_rays, cfg.train.chunk_size)
	coarse_rgb_maps_list = []
	fine_rgb_maps_list = []
	for chunk in chunks:
		
		# Retrieve the fields from concatenated batch
		ray_origins_batch, ray_directions_batch, cone_radii_batch = chunk[...,:3], chunk[...,3:6], chunk[..., 6]
		if cfg.model.use_viewdirs:
			viewdirs_batch = chunk[..., 7:]

		################# Rendering of 3D coarse points for the ray batch #################
	    # Perform stratified sampling of rays to generate coarse points
		coarse_depth_values = get_coarse_depth_values(ray_directions_batch, ray_origins_batch,
                                                        near_thresh, far_thresh, cfg)
		coarse_sample_means, coarse_sample_covs = sample_points(ray_directions_batch, ray_origins_batch, coarse_depth_values, cone_radii_batch)
		
        # Pass the coarse points to the NN model to predict RGBA values for the points
		if not cfg.model.use_viewdirs:
			viewdirs_batch = None
		rgba_coarse = get_radiance_field_per_chunk_mip(coarse_sample_means, coarse_sample_covs, model, viewdirs_batch, cfg)
		
		# TODO: 1. Check extra arguments - white_bckgd, noise
		# TODO: 2. shud we mutiply by ray_directions ?
		
        # Computes rgb values of the image based on the likelihood of sample points and their corresponding 3D info like RGB, density, depth etc
		rgb_map_coarse, _, _, _, weights = render_image_batch_from_3dinfo_mip(rgba_coarse , coarse_depth_values, cfg) 
		coarse_rgb_maps_list.append(rgb_map_coarse)
		
        ################# Rendering of 3D fine points for the ray batch #################
		if cfg.model.num_fine_samples_per_ray > 0:
			
            # TODO: check this and change these variable names aptly
			weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
			weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
			weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
			resample_padding = 0.01
			weights = weights_blur + resample_padding
	
			depth_values_mid = .5 * (coarse_depth_values[..., 1:] + coarse_depth_values[..., :-1])
			z_samples = sample_pdf(depth_values_mid, weights[..., 1:-1], cfg.model.num_fine_samples_per_ray, True)
			z_samples = z_samples.detach()
			fine_depth_values, _ = torch.sort(torch.cat((coarse_depth_values, z_samples), dim=-1), dim=-1)
			fine_sample_means, fine_sample_covs = sample_points(ray_directions_batch, ray_origins_batch, fine_depth_values, cone_radii_batch)

            # Pass the fine points to the NN model to predict RGBA values for the points
			rgba_fine = get_radiance_field_per_chunk_mip(fine_sample_means, fine_sample_covs, model, viewdirs_batch, cfg)

			# Computes rgb values of the image based on the likelihood of sample points and their corresponding 3D info like RGB, density, depth etc
			rgb_map_fine, _, _, _, _ = render_image_batch_from_3dinfo_mip(rgba_fine , fine_depth_values, cfg) 
			fine_rgb_maps_list.append(rgb_map_fine)

    # Convert the coarse and fine RGB lists to tensors
	rgb_coarse_image = torch.cat(coarse_rgb_maps_list, dim=0)
	rgb_fine_image = torch.cat(fine_rgb_maps_list, dim=0)

	return rgb_coarse_image, rgb_fine_image, selected_ray_coors