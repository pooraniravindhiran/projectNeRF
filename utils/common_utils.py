import torch
import math

def get_chunks(tensor: torch.Tensor, chunk_size: int):
  chunks = [tensor[i:i+chunk_size] for i in range(0, tensor.shape[0], chunk_size)]
  return chunks

def mse2psnr(mse):
  return -10. * math.log10(mse)

def cumprod_exclusive(tensor: torch.Tensor):
  # for input (a,b,c), cumprod_inclusive is (a, a*b, a*b*c)
  cumprod_inclusive = torch.cumprod(tensor, dim=-1)

  # for input (a,b,c), cumprod_exclusive is (1, a, a*b)
  cumprod_exclusive = torch.roll(cumprod_inclusive, 1, dims=-1) # (a*b*c, a, a*b)
  cumprod_exclusive[..., 0] = 1.

  return cumprod_exclusive

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
    
    # To prevent nans in weights
    weights = weights + 1e-5
    
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