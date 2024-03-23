import torch

def get_params_of_gaussian_cone_coors(depth_values, cone_radii):
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

def get_params_of_gaussian_world_coors(mu_t, sigma_t, sigma_r, ray_origins, ray_directions):
    means = ray_origins[...,None,:] + mu_t[...,None]*ray_directions[...,None,:]

    dir_mag_sq =  torch.sum(ray_directions ** 2, dim=-1).unsqueeze(dim=-1)
    covars = ( sigma_t[...,None] * ((ray_directions ** 2)[...,None,:]) + \
    (sigma_r[...,None] * (1. - (ray_directions ** 2)/dir_mag_sq)[...,None,:]) )

    return means, covars

# Sample 3D points and their depth values on the ray bundle
# sampled 3d points will be of shape h x w x num_samples x 3 and depth_values will be of shape h x w x num_samples
# TODO: t_values vs depth_values?
def sample_coarse_points_mip(ray_directions: torch.Tensor, ray_origins: torch.Tensor, nearThresh: float, farThresh: float, \
                         cone_radii: torch.Tensor, num_points: int, linear_disparity):
  t_values = torch.linspace(0., 1., num_points+1).to(device) # (num_points)
  if linear_disparity:
    z_values = 1. / (1. / nearThresh * (1. - t_values) + 1. / farThresh * t_values)
  else:
    z_values = nearThresh * (1. - t_values) + farThresh * t_values
  z_values = z_values.expand([ray_directions.shape[0], num_points+1])


  # Randomize depth values
  mids = 0.5 * (z_values[..., 1:] + z_values[..., :-1])
  upper = torch.cat((mids, z_values[..., -1:]), dim=-1)
  lower = torch.cat((z_values[..., :1], mids), dim=-1)
  t_rand = torch.rand(z_values.shape).to(device)
  depth_values = lower + (upper - lower) * t_rand

  # sample_points = ray_origins[...,None,:] + (ray_directions[...,None,:] * depth_values[...,:,None])
  mu_t, sigma_t, sigma_r = get_params_of_gaussian_cone_coors(depth_values, cone_radii)
  means, covars  = get_params_of_gaussian_world_coors(mu_t, sigma_t, sigma_r, ray_origins, ray_directions) # chunk * num_encoding * 3
  # print(means.shape, covars.shape)
  return depth_values, means, covars

def render_image_batch_from_3dinfo(rgb_density: torch.Tensor, depth_values: torch.Tensor, use_white_bkgd: bool):

  # Normalize RGB values to the range 0 to 1 and since density (opacity/ amount of light absorbed/ absorption coeff) is non-negative, apply relu activation.
  rgb_values = torch.sigmoid(rgb_density[..., :3]) # (chunk_size x num_samples x 3)
  density_values = torch.nn.functional.relu(rgb_density[..., 3])

  # print(rgb_values.shape, density_values.shape)

  # Compute consecutive difference between depth values, i.e distance of every segment sampled in 3D space
  dist_values = depth_values[...,1:]-depth_values[...,:-1]

  # For the last segment i.e between the last sampled point and ray end, distance is unbounded. So to make it tractable, we add last segment distance to a large value.
  one_e10 = torch.tensor([1e10], dtype=depth_values.dtype, device=depth_values.device).expand(depth_values[..., :1].shape)
  # dist_values = torch.cat([dist_values, one_e10], dim=-1) # (h x w x num_samples)
  # print(dist_values.shape)
  # Compute cumulative transmittance along every segment (complement of absorption, i.e amount of light trasmitted through medium after absorption)

  transmittance_values = 1. - torch.exp(-density_values * dist_values)
  num_stability_val = 1e-10
  cum_transmittance_values = transmittance_values * cumprod_exclusive(1. - transmittance_values + num_stability_val) # (h x w x num_samples)
  # TODO: why 1-transmittance values

  # Render RGB map i.e RGB image
  rgb_map = (cum_transmittance_values[..., None] * rgb_values).sum(dim=-2) # (h x w x 3)
  # Compute cum trasnmittance for image
  accumulated_transmittance_map = cum_transmittance_values.sum(dim=-1)

  # Render depth map i.e. depth image # TODO CHECK
  t_mids = 0.5 * (depth_values[..., :-1] + depth_values[..., 1:])
  distance = (cum_transmittance_values * t_mids).sum(dim=-1) / accumulated_transmittance_map
  depth_map = torch.clamp(torch.nan_to_num(distance), depth_values[:, 0], depth_values[:, -1])
  # depth_map = (cum_transmittance_values * depth_values).sum(dim=-1) # (h x w x 1)



  # White background
  if use_white_bkgd:
    rgb_map = rgb_map + (1. - accumulated_transmittance_map[...,None])

  # Disparity Map
  disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / accumulated_transmittance_map)

  return rgb_map, disp_map, accumulated_transmittance_map, depth_map, cum_transmittance_values

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