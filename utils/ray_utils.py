import torch
import numpy as np
import torch.nn as nn
from utils import cumprod_exclusive

#TODO : device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Input is of shape (chunk_size x num_samples x 4)
# Output is of shape (chunk_size x 3)
# TODO: Add diagramatic explanation for understanding
def render_image_batch_from_3dinfo(rgb_density: torch.Tensor, depth_values: torch.Tensor, use_white_bkgd = False):

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

    # White background
    if use_white_bkgd:
        rgb_map = rgb_map + (1. - accumulated_transmittance_map[...,None])

    # Disparity Map
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / accumulated_transmittance_map)

    return rgb_map, disp_map, accumulated_transmittance_map, depth_map, cum_transmittance_values