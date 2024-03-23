import torch

import numpy as np
import torch.nn as nn

from utils.common_utils import *

def get_raybundle_for_img(height: int, width: int, focal_length: float, tf_cam2world: torch.Tensor, device: torch.device):
    """
    Computes directions and origins for rays for all pixels in the image.

    Args:
        height (int): Height of the image.
        width (int): Width of the image.
        focal_length (float): Focal length of the camera.
        tf_cam2world (torch.Tensor): Transformation matrix from camera to world coordinates or pose of camera.

    Returns:
        ray_origins (torch.Tensor): Ray origins for all pixels in world coordinates. Shape: (height, width, 3).
        ray_directions (torch.Tensor): Ray directions for all pixels in world coordinates. Shape: (height, width, 3).
    """

    # Create a meshgrid for the image pixels
    pixel_coors_along_w = torch.arange(width)
    pixel_coors_along_h = torch.arange(height)
    wcoor_grid_pixel = pixel_coors_along_w.expand(height,-1).to(device)
    hcoor_grid_pixel = pixel_coors_along_h.view(-1,1).expand(-1, width).to(device)

    # Compute ray directions extending from optical center to 3d world for every pixel.
    # This is basically expressing pixel coordinates in world frame.
    wcoor_grid_cam = (wcoor_grid_pixel - (width/2))/ focal_length
    hcoor_grid_cam = -1*(hcoor_grid_pixel - (height/2))/ focal_length
    pixel_coors_cam = torch.stack([wcoor_grid_cam, hcoor_grid_cam, -torch.ones_like(wcoor_grid_cam)], dim=-1)

    # To convert cam to world frame, 3x3 rotation matrix of pose must be multiplied by column vector of each point.
    # We have hxwx3 matrix. Write down the multiplication for one element and you'll know how to implement it.
    pixel_coors_world = torch.sum(pixel_coors_cam[..., None, :] * tf_cam2world[:3, :3], dim=-1)
    ray_directions_world = pixel_coors_world

    ray_origins_world = tf_cam2world[:3,-1].expand(pixel_coors_world.shape)

    return ray_origins_world.double(), ray_directions_world.double()

def tf_world2ndc(ray_origins_world: torch.Tensor, ray_directions_world: torch.Tensor, near: float, width: int, 
                 height: int, focal_length: float):
    """
    Converts world coordinates to Normalized Device Coordinates (NDC) so that in forward-facing captures the scene can be bounded.

    Notes:
    In world space, depth values are sampled between `near` and `f`, where `f` is infinity in forward-facing scenes.
    In NDC space, depth values are sampled between `near` and 1. 
    After shifting the origin to the near plane, depth values are sampled between 0 and 1.

    Args:
        ray_origins_world (torch.Tensor): Ray origins in world coordinates. Shape: (height, width, 3).
        ray_directions_world (torch.Tensor): Ray directions in world coordinates. Shape: (height, width, 3).
        near (float): Near plane distance.
        width (int): Width of the image.
        height (int): Height of the image.
        focal_length (float): Focal length of the camera.

    Returns:
        torch.Tensor: Ray origins in NDC. Shape: (height, width, 3).
        torch.Tensor: Ray directions in NDC. Shape: (height, width, 3).
     """

    # Shift ray origins to near plane
    depths = -(near + ray_origins_world[..., 2])/ ray_directions_world[..., 2]
    ray_origins_world = ray_origins_world + depths[..., None] * ray_directions_world # TODO- check why None

    # Project from world to ndc space. Formulas obtained from NeRF paper.
    o_x = (-focal_length * ray_origins_world[...,0]) / ((width/2.)*ray_origins_world[...,2])
    o_y = (-focal_length * ray_origins_world[...,1]) / ((height/2.)*ray_origins_world[...,2])
    o_z = ((2.0*focal_length) + ray_origins_world[...,2]) / (ray_origins_world[...,2])
    d_x = (-focal_length / (width/2.0)) * ((ray_directions_world[...,0]/ray_directions_world[...,2])-(ray_origins_world[...,0]/ray_origins_world[...,2]))
    d_y = (-focal_length / (height/2.0)) * ((ray_directions_world[...,1]/ray_directions_world[...,2])-(ray_origins_world[...,1]/ray_origins_world[...,2]))
    d_z = (-2.0 * focal_length) / (ray_origins_world[...,2])

    ray_origins_ndc = torch.stack([o_x, o_y, o_z], -1)
    ray_directions_ndc = torch.stack([d_x, d_y, d_z], -1)
    return ray_origins_ndc, ray_directions_ndc

def sample_coarse_points(ray_directions: torch.Tensor, ray_origins: torch.Tensor, nearThresh: float, farThresh: float, cfg):
    """
    Samples 3D coarse points along the rays defined by directions and origins.

    Args:
        ray_directions (torch.Tensor): Tensor representing ray directions. Shape: (x, 3).
        ray_origins (torch.Tensor): Tensor representing ray origins. Shape: (x, 3).
        nearThresh (float): Near threshold for sampling.
        farThresh (float): Far threshold for sampling.
        cfg: dictionary like object representing user configuration.

    Returns:
        torch.Tensor: Depth values of sampled points along each ray. Shape: (x, num_points).
        torch.Tensor: Sampled points along each ray. Shape: (x, num_points, 3).
    """
    # TODO: t_values vs depth_values?
    t_values = torch.linspace(0., 1., cfg.model.num_coarse_samples_per_ray).to(cfg.device) # (num_points)

    # Check if linear disparity needs to be applied
    if cfg.dataset.is_ndc_required:
        z_values = 1. / (1. / nearThresh * (1. - t_values) + 1. / farThresh * t_values)
    else:
        z_values = nearThresh * (1. - t_values) + farThresh * t_values

    z_values = z_values.expand([ray_directions.shape[0], cfg.model.num_coarse_samples_per_ray])

    # Randomize depth values
    mids = 0.5 * (z_values[..., 1:] + z_values[..., :-1])
    upper = torch.cat((mids, z_values[..., -1:]), dim=-1)
    lower = torch.cat((z_values[..., :1], mids), dim=-1)
    t_rand = torch.rand(z_values.shape).to(cfg.device)
    depth_values = lower + (upper - lower) * t_rand

    sample_points = ray_origins[...,None,:] + (ray_directions[...,None,:] * depth_values[...,:,None])
    return depth_values, sample_points

def get_radiance_field_per_chunk(sample_points, model, viewdirs_batch, cfg):
    """
    Computes radiance field (RGBD values) per chunk of sample points using the given model.

    Args:
        sample_points (torch.Tensor): Tensor representing the sample points. Shape: (h * w, num_samples, 3).
        model (torch.nn.Module): Neural network model to compute radiance field.
        viewdirs_batch (torch.Tensor): Tensor representing the directional encoding batch.
        cfg: dictionary like object representing user configuration.

    Returns:
        torch.Tensor: Tensor representing the radiance field computed per chunk. Shape: (h, w, num_samples, 4).
    """
	
    # Flatten sample points from shape (x, num_points, 3) into shape (x*num_points, 3)
    flattened_sample_points = sample_points.view(-1,3) # (h*w*num, 3)

	# Encode sample points using positional embedding
    encoded_sample_points = perform_positional_encoding(flattened_sample_points, cfg.model.num_pos_encoding_func, cfg.model.include_input_in_posencoding)
    if cfg.model.use_viewdirs:
        ipdirs_batch = viewdirs_batch[...,None,:].expand(sample_points.shape) # (h*w,num,3)
        ipdirs_batch = ipdirs_batch.reshape(-1, 3) # (h*w*num, 3)
        encoded_dirs = perform_positional_encoding(ipdirs_batch, cfg.model.num_dir_encoding_func, cfg.model.include_input_in_direncoding)
        encoded_sample_points_batch = torch.cat((encoded_sample_points, encoded_dirs), dim=-1)

    # Batchify and call NN model on the batch
    chunks = get_chunks(encoded_sample_points_batch, cfg.train.chunk_size)
    rgba_list = []
    for chunk in chunks:
        rgba_list.append(model(chunk.to(cfg.device)))
    rgba_list = torch.cat(rgba_list, dim=0)
    rgba_list = rgba_list.reshape(list(sample_points.shape[:-1]) + [rgba_list.shape[-1]])
        
    return rgba_list

def render_image_batch_from_3dinfo(rgb_density: torch.Tensor, depth_values: torch.Tensor, cfg):
    """
    Renders an image batch from 3D information including RGB values, depth values, and density values. 

    Args:
        rgb_density (torch.Tensor): Tensor representing RGB and density information. Shape: (chunk_size x num_samples x 4).
        depth_values (torch.Tensor): Tensor representing depth values. Shape: (chunk_size x num_samples).
        cfg: dictionary like object representing user configuration.

    Returns:
        torch.Tensor: Rendered RGB map. Shape: (chunk_size x 3).
        torch.Tensor: Disparity map i.e inverse of depth map. Shape: (chunk_size x 1).
        torch.Tensor: Accumulated transmittance map. Shape: (chunk_size x 1).
        torch.Tensor: Depth map. Shape: (chunk_size x 1).
        torch.Tensor: Cumulative transmittance values. Shape: (chunk_size x num_samples).
    """

    # Normalize RGB values to the range 0 to 1 by applying sigmoid activation
    rgb_values = torch.sigmoid(rgb_density[..., :3]) # (chunk_size x num_samples x 3)

    # Since density (opacity/ amount of light absorbed/ absorption coeff) is non-negative, apply relu activation for density
    density_values = torch.nn.functional.relu(rgb_density[..., 3])

    # Compute consecutive difference between depth values, i.e distance of every segment sampled in 3D space
    dist_values = depth_values[...,1:]-depth_values[...,:-1]

    # For the last segment i.e between the last sampled point and ray end, distance is unbounded. 
    # So to make it tractable, we add last segment distance to a large value.
    one_e10 = torch.tensor([1e10], dtype=depth_values.dtype, device=depth_values.device).expand(depth_values[..., :1].shape)
    dist_values = torch.cat([dist_values, one_e10], dim=-1) # (h x w x num_samples)

    # Compute cumulative transmittance along every segment 
    # This is complement of absorption, i.e amount of light trasmitted through medium after absorption
    transmittance_values = 1. - torch.exp(-density_values * dist_values)
    num_stability_val = 1e-10
    cum_transmittance_values = transmittance_values * compute_cumprod_exclusive(1. - transmittance_values + num_stability_val) # (h x w x num_samples)
    # TODO: why 1-transmittance values

    # Render RGB map i.e RGB image
    rgb_map = (cum_transmittance_values[..., None] * rgb_values).sum(dim=-2) # (h x w x 3)

    # Render depth map i.e. depth image
    depth_map = (cum_transmittance_values * depth_values).sum(dim=-1) # (h x w x 1)

    # Compute cum transmittance for image
    accumulated_transmittance_map = cum_transmittance_values.sum(dim=-1)

    # Check if user wants a white background
    if cfg.dataset.use_white_bkgd:
        rgb_map = rgb_map + (1. - accumulated_transmittance_map[...,None])

    # Generate disparity map # TODO: check what is this and how it differs between depth map
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / accumulated_transmittance_map)

    return rgb_map, disp_map, accumulated_transmittance_map, depth_map, cum_transmittance_values
