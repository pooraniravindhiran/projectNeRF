import torch

from utils.common_utils import *

# TODO: add func descriptions and make it similar to ray utils

def get_params_of_gaussian_cone_coors(depth_values: torch.Tensor, cone_radii: torch.Tensor):
    t0 = depth_values[..., :-1]
    t1 = depth_values[..., 1:]
    cone_radii = cone_radii.unsqueeze(dim=-1)
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                        (3 * mu**2 + hw**2)**2)
    r_var = cone_radii**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                (hw**4) / (3 * mu**2 + hw**2))
    return t_mean, t_var, r_var

def get_params_of_gaussian_world_coors(mu_t: torch.Tensor, sigma_t: torch.Tensor, sigma_r: torch.Tensor, 
                                       ray_origins: torch.Tensor, ray_directions: torch.Tensor):
    means = ray_origins[...,None,:] + mu_t[...,None]*ray_directions[...,None,:]

    dir_mag_sq =  torch.sum(ray_directions ** 2, dim=-1).unsqueeze(dim=-1)
    covars = ( sigma_t[...,None] * ((ray_directions ** 2)[...,None,:]) + \
    (sigma_r[...,None] * (1. - (ray_directions ** 2)/dir_mag_sq)[...,None,:]) )

    return means, covars

def get_coarse_depth_values(ray_directions: torch.Tensor, ray_origins: torch.Tensor, nearThresh: float, farThresh: float, \
                                cfg):
    t_values = torch.linspace(0., 1., cfg.model.num_coarse_samples_per_ray+1).to(cfg.device) # (num_points)
    if cfg.dataset.is_ndc_required:
        z_values = 1. / (1. / nearThresh * (1. - t_values) + 1. / farThresh * t_values)
    else:
        z_values = nearThresh * (1. - t_values) + farThresh * t_values
    z_values = z_values.expand([ray_directions.shape[0], cfg.model.num_coarse_samples_per_ray+1])

    # Randomize depth values
    mids = 0.5 * (z_values[..., 1:] + z_values[..., :-1])
    upper = torch.cat((mids, z_values[..., -1:]), dim=-1)
    lower = torch.cat((z_values[..., :1], mids), dim=-1)
    t_rand = torch.rand(z_values.shape).to(cfg.device)
    depth_values = lower + (upper - lower) * t_rand

    return depth_values

def sample_points(ray_directions: torch.Tensor, ray_origins: torch.Tensor, depth_values: torch.Tensor, cone_radii: torch.Tensor):
    mu_t, sigma_t, sigma_r = get_params_of_gaussian_cone_coors(depth_values, cone_radii)
    means, covars  = get_params_of_gaussian_world_coors(mu_t, sigma_t, sigma_r, ray_origins, ray_directions)

    return means, covars

def get_radiance_field_per_chunk_mip(mu_tensor, diag_sigma_tensor, model, 
                                     viewdirs_batch, cfg):
	
    # Encode sample points using integrated positional embedding
    encoded_sample_points = perform_integrated_positional_encoding(mu_tensor, diag_sigma_tensor, 
                                                                   cfg.model.num_pos_encoding_func)
    # encoded_sample_points = positional_encoding_obj(mu_tensor, diag_sigma_tensor)[0]
    encoded_sample_points = encoded_sample_points.reshape(-1, encoded_sample_points.shape[-1])

    if cfg.model.use_viewdirs:
        ipdirs_batch = viewdirs_batch[...,None,:].expand(mu_tensor.shape) # h*w,num,3
        ipdirs_batch = ipdirs_batch.reshape(-1, 3) # h*w*num, 3
        # encoded_dirs = viewdirs_encoding_obj(ipdirs_batch.to(device))
        encoded_dirs = perform_positional_encoding(ipdirs_batch, cfg.model.num_dir_encoding_func, 
                                                    cfg.model.include_input_in_direncoding)
        encoded_sample_points = torch.cat((encoded_sample_points, encoded_dirs), dim=-1)

    # Batchify and call NN model on the batch
    chunks = get_chunks(encoded_sample_points, cfg.train.chunk_size)
    rgba_list = []
    for chunk in chunks:
        rgba_list.append(model(chunk.to(cfg.device)))
    rgba_list = torch.cat(rgba_list, dim=0)
    rgba_list = rgba_list.reshape(list(sample_points.shape[:-1]) + [rgba_list.shape[-1]])
        
    return rgba_list

def get_radiance_field_per_chunk_mip(mu_tensor, diag_sigma_tensor, model, 
                                     viewdirs_batch, cfg):

    # Encode sample points using integrated positional embedding
    encoded_sample_points = perform_integrated_positional_encoding(cfg.model.num_pos_encoding_func,
                                                                   cfg.device, mu_tensor, diag_sigma_tensor)
    encoded_sample_points = encoded_sample_points.reshape(-1, encoded_sample_points.shape[-1])

    if cfg.model.use_viewdirs:
            ipdirs_batch = viewdirs_batch[...,None,:].expand(mu_tensor.shape) # h*w,num,3
            ipdirs_batch = ipdirs_batch.reshape(-1, 3) # h*w*num, 3
            encoded_dirs = perform_integrated_positional_encoding(cfg.model.num_dir_encoding_func, 
                                                       cfg.device, ipdirs_batch)
            encoded_sample_points = torch.cat((encoded_sample_points, encoded_dirs), dim=-1)
            
    # Batchify
    ip_chunks = get_chunks(encoded_sample_points, cfg.train.chunk_size)
    rgba_batch = []
    for chunk in ip_chunks:
        rgba_batch.append(model(chunk.to(cfg.device)))

    rgba_batch = torch.cat(rgba_batch, dim=0)
    rgba_batch = rgba_batch.reshape(list(mu_tensor.shape[:-1]) + [rgba_batch.shape[-1]])
    return rgba_batch

def render_image_batch_from_3dinfo_mip(rgb_density: torch.Tensor, depth_values: torch.Tensor, cfg):
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

    # Compute cumulative transmittance along every segment 
    # This is complement of absorption, i.e amount of light trasmitted through medium after absorption
    transmittance_values = 1. - torch.exp(-density_values * dist_values)
    num_stability_val = 1e-10
    cum_transmittance_values = transmittance_values * compute_cumprod_exclusive(1. - transmittance_values + num_stability_val) # (h x w x num_samples)
    # TODO: why 1-transmittance values

    # Render RGB map i.e RGB image
    rgb_map = (cum_transmittance_values[..., None] * rgb_values).sum(dim=-2) # (h x w x 3)

    # Compute cum transmittance for image
    accumulated_transmittance_map = cum_transmittance_values.sum(dim=-1)

    # Render depth map i.e. depth image
    t_mids = 0.5 * (depth_values[..., :-1] + depth_values[..., 1:])
    distance = (cum_transmittance_values * t_mids).sum(dim=-1) / accumulated_transmittance_map
    depth_map = torch.clamp(torch.nan_to_num(distance), depth_values[:, 0], depth_values[:, -1]) # (h x w x 1)

    # Check if user wants a white background
    if cfg.dataset.use_white_bkgd:
        rgb_map = rgb_map + (1. - accumulated_transmittance_map[...,None])

    # Generate disparity map # TODO: check what is this and how it differs between depth map
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / accumulated_transmittance_map)

    return rgb_map, disp_map, accumulated_transmittance_map, depth_map, cum_transmittance_values
