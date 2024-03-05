import torch
import numpy as np
import torch.nn as nn
from utils import cumprod_exclusive

#TODO : device 

# Compute directions and origins for rays for all pixels in the image.
# Both will be of shape  h x w x 3
def get_raybundle_for_img(height: int, width: int, focal_length: float, tf_cam2world: torch.Tensor):

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

    return ray_origins_world, ray_directions_world

# Converts world coordinates to Normalized Device coordinates so that in forward-facing captures the scene can be bounded.
# In world space, you will sample depth values between n and f, where f=infinity in forward-facing scenes.
# In ndc space, you will instead sample depth values between n and 1. After shifting origin to near plane, you can sample depth values between 0 and 1.
def tf_world2ndc(ray_origins_world, ray_directions_world, near, width, height, focal_length):
    # Shift ray origins to near plane
    depths = -(near + ray_origins_world[..., 2])/ ray_directions_world[..., 2]
    ray_origins_world = ray_origins_world + depths[..., None] * ray_directions_world # TODO- check why None

    # Project from world to ndc space
    o_x = (-focal_length * ray_origins_world[...,0]) / ((width/2.)*ray_origins_world[...,2])
    o_y = (-focal_length * ray_origins_world[...,1]) / ((height/2.)*ray_origins_world[...,2])
    o_z = ((2.0*focal_length) + ray_origins_world[...,2]) / (ray_origins_world[...,2])
    d_x = (-focal_length / (width/2.0)) * ((ray_directions_world[...,0]/ray_directions_world[...,2])-(ray_origins_world[...,0]/ray_origins_world[...,2]))
    d_y = (-focal_length / (height/2.0)) * ((ray_directions_world[...,1]/ray_directions_world[...,2])-(ray_origins_world[...,1]/ray_origins_world[...,2]))
    d_z = (-2.0 * focal_length) / (ray_origins_world[...,2])

    ray_origins_ndc = torch.stack([o_x, o_y, o_z], -1)
    ray_directions_ndc = torch.stack([d_x, d_y, d_z], -1)
    return ray_origins_ndc, ray_directions_ndc


# Sample 3D points and their depth values on the ray bundle
# sampled 3d points will be of shape h x w x num_samples x 3 and depth_values will be of shape h x w x num_samples
# TODO: t_values vs depth_values?
def sample_coarse_points(ray_directions: torch.Tensor, ray_origins: torch.Tensor, nearThresh: float, farThresh: float, num_points: int, linear_disparity):
    t_values = torch.linspace(0., 1., num_points).to(device) # (num_points)
    if linear_disparity:
        z_values = 1. / (1. / nearThresh * (1. - t_values) + 1. / farThresh * t_values)
    else:
        z_values = nearThresh * (1. - t_values) + farThresh * t_values
    z_values = z_values.expand([ray_directions.shape[0], num_points])

    # Randomize depth values
    mids = 0.5 * (z_values[..., 1:] + z_values[..., :-1])
    upper = torch.cat((mids, z_values[..., -1:]), dim=-1)
    lower = torch.cat((z_values[..., :1], mids), dim=-1)
    t_rand = torch.rand(z_values.shape).to(device)
    depth_values = lower + (upper - lower) * t_rand

    sample_points = ray_origins[...,None,:] + (ray_directions[...,None,:] * depth_values[...,:,None])
    return depth_values, sample_points

# Perform positional encoding on a tensor to get a high-dimensional representation enabling better capture of high frequency variations and
# to capture the relationship between tensor values.
# encoding is of shape (h * w * num_samples, 3 +(2 * num_encoding_functions * 3))
def positional_encoding(tensor: torch.Tensor, num_encoding_functions: int, include_input: bool):
    if include_input:
        encoding = [tensor] # (h * w * num_samples, 3)
    else:
        encoding = []

    frequency_band = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
    )
    for frequency in frequency_band:
        for func in [torch.sin, torch.cos]:
            sinusoidal_component = func(tensor * frequency)
            encoding.append(sinusoidal_component)

    return torch.cat(encoding, dim=-1)


# Do inverse tranform sampling- https://en.wikipedia.org/wiki/Inverse_transform_sampling
def sample_pdf(bins, weights, num_samples, sample_randomly):

    # Calculate PDF. Divide by sum to get them between 0 and 1.
    pdf = weights/weights.sum(-1).unsqueeze(-1)

    # Compute CDF for all bins starting from 0 to 1
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[...,:1]), cdf), dim=-1)

    # Generate samples from standard uniform distribution deterministically or randomly
    if sample_randomly:
        samples_uniform = torch.rand(list(cdf.shape[:-1])+[num_samples]).to(weights)
    else:
        samples_uniform = torch.linspace(0., 1., num_samples).to(weights)
        samples_uniform = samples_uniform.expand(list(cdf.shape[:-1])+[num_samples])

    # Find indices where cdf value is just above the sampled value
    indices = torch.searchsorted(cdf, samples_uniform, side='right')

    # Ensure they are within the range of possible indices and find the indices that are right below and above these
    below_indices = torch.max(torch.zeros_like(indices), indices-1)
    above_indices = torch.min((cdf.shape[-1]-1) * torch.ones_like(indices) , indices)

    # Find the respective cdf values and the bin values
    cdf_for_below_indices = torch.gather(cdf, -1, below_indices)
    bin_for_below_indices = torch.gather(bins, -1, below_indices)
    cdf_for_above_indices = torch.gather(cdf, -1, above_indices)
    bin_for_above_indices = torch.gather(bins, -1, above_indices)

    # Find denom and ensure numerical stability during division
    denom = cdf_for_above_indices - cdf_for_below_indices
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)

    # Find interpolation weights
    # If sample_uniform is close to below index, then interpolation weight will be close to 0.
    interpolation_weights = (samples_uniform - cdf_for_below_indices)/denom

    # Perform linear interpolation
    # if interpolation weight = 0, then below will be chosen.
    # if interpolation weight = 1, then above bin will be chosen.
    # if interpolation weight is between 0 and 1, linear interpolation.
    samples_cdf = bin_for_below_indices + (interpolation_weights * (bin_for_above_indices-bin_for_below_indices))

    return samples_cdf


# Input is of shape (chunk_size x num_samples x 4)
# Output is of shape (chunk_size x 3)
# TODO: Add diagramatic explanation for understanding
def render_image_batch_from_3dinfo(rgb_density: torch.Tensor, depth_values: torch.Tensor):

    # Normalize RGB values to the range 0 to 1 and since density (opacity/ amount of light absorbed/ absorption coeff) is non-negative, apply relu activation.
    rgb_values = torch.sigmoid(rgb_density[..., :3]) # (chunk_size x num_samples x 3)
    density_values = torch.nn.functional.relu(rgb_density[..., 3])

    # print(rgb_values.shape, density_values.shape)

    # Compute consecutive difference between depth values, i.e distance of every segment sampled in 3D space
    dist_values = depth_values[...,1:]-depth_values[...,:-1]

    # For the last segment i.e between the last sampled point and ray end, distance is unbounded. So to make it tractable, we add last segment distance to a large value.
    one_e10 = torch.tensor([1e10], dtype=depth_values.dtype, device=depth_values.device).expand(depth_values[..., :1].shape)
    dist_values = torch.cat([dist_values, one_e10], dim=-1) # (h x w x num_samples)
    # print(dist_values.shape)
    # Compute cumulative transmittance along every segment (complement of absorption, i.e amount of light trasmitted through medium after absorption)
    transmittance_values = 1. - torch.exp(-density_values * dist_values)
    num_stability_val = 1e-10
    cum_transmittance_values = transmittance_values * cumprod_exclusive(1. - transmittance_values + num_stability_val) # (h x w x num_samples)
    # TODO: why 1-transmittance values

    # Render RGB map i.e RGB image
    rgb_map = (cum_transmittance_values[..., None] * rgb_values).sum(dim=-2) # (h x w x 3)

    # Render depth map i.e. depth image
    depth_map = (cum_transmittance_values * depth_values).sum(dim=-1) # (h x w x 1)

    # Compute cum trasnmittance for image
    accumulated_transmittance_map = cum_transmittance_values.sum(dim=-1)

    # Disparity Map
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / accumulated_transmittance_map)

    return rgb_map, disp_map, accumulated_transmittance_map, depth_map, cum_transmittance_values

