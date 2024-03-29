import torch
import torchvision

import math
import yaml
import os
import numpy as np
from easydict import EasyDict
from skimage.metrics import structural_similarity

def read_config(filename: str):
    """
    Read a YAMl file and return its content as a dictionary like object.
    
    Args:
        filename (str): The path to the YAML file.
    
    Returns: 
        config (easydict): The content of the YAMl file as a dictionary object.
    
    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        RunTimeError: If an error occurs while reading the YAML file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    try:
        with open(filename, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
    except (IOError, OSError, yaml.YAMLError) as e:
        raise RuntimeError(f"Error reading the config YAML file: {e}")
    return config

def cast_tensor_to_image(tensor: torch.Tensor):
    """
    Converts a tensor to PIL image suitable for visualization in TensorBoard.

    Args:
        tensor (torch.Tensor): Input tensor representing an image with shape (H, W, 3).

    Returns:
        numpy.ndarray: Numpy array representing the image with shape (3, H, W), suitable for TensorBoard visualization.
    """
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)

    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    img = np.moveaxis(img, [-1], [0])

    return img

def get_chunks(tensor: torch.Tensor, chunk_size: int):
    """
    Splits a given tensor into batches/ chunks along its first dimension.

    Args:
        tensor (torch.Tensor): The input tensor to be split into batches.
        chunk_size (int): The size of each batch.

    Returns:
        chunks (torch.Tensor): A list of tensors, each containing a chunk of the input tensor.
    """
    chunks = [tensor[i:i+chunk_size] for i in range(0, tensor.shape[0], chunk_size)]
    return chunks

def compute_ssim_score(predicted_img: torch.Tensor, target_img: torch.Tensor):
    """
    Computes the Structural Similarity Index (SSIM) between a predicted image and a target image.

    Args:
        predicted_img (torch.Tensor): The predicted image tensor.
        target_img (torch.Tensor): The target image tensor.

    Returns:
        score (float): The SSIM score between the predicted and target images.
    """
    score = structural_similarity(predicted_img, target_img, channel_axis=0, full=False)
    return score

def convert_mse_to_psnr(mse: float):
    """
    Convert Mean Squared Error (MSE) to Peak Signal-to-Noise Ratio (PSNR) in decibels.

    Args:
        mse (float): Mean Squared Error (MSE) value.

    Returns:
        float: Peak Signal-to-Noise Ratio (PSNR) value in decibels.
    """
    return -10. * math.log10(mse)

def compute_cumprod_exclusive(tensor: torch.Tensor):
    """
    Computes the exclusive cumulative product along the last dimension of a tensor.
    Example:
        tensor = torch.tensor([1, 2, 3, 4])
        result = cumprod_exclusive(tensor)
        print(result)
        tensor([1., 1., 2., 6.])

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        cumprod_exclusive (torch.Tensor): A tensor representing the exclusive cumulative product along the last dimension of the input tensor.
    """
    # For input [a,b,c]: cumprod_inclusive is [a, a*b, a*b*c]
    cumprod_inclusive = torch.cumprod(tensor, dim=-1)

    # For input [a,b,c]: cumprod_exclusive is [1, a, a*b]
    cumprod_exclusive = torch.roll(cumprod_inclusive, 1, dims=-1) # [a*b*c, a, a*b]
    cumprod_exclusive[..., 0] = 1.

    return cumprod_exclusive

def perform_positional_encoding(num_encoding_func: int, device: torch.device, tensor: torch.Tensor):
    """
    Encodes tensor with sinusoidal components for different frequency bands based on the number of encoding functions specified.
    This helps to get a high-dimensional representation of the input, enabling better capture of high frequency variations and
    to capture the relationship between tensor values.

    Args:
        tensor (torch.Tensor): The input tensor of shape (x, 3)
        num_encoding_func (int): The number of encoding functions to use.
        include_input (bool): Whether to include the input tensor in the encoding.

    Returns:
        torch.Tensor: A tensor representing the positional encoding
            If include_input is False, it is of shape (x, 2*num_encoding_func*3) 
            Else, it is of shape (x, 3 + (2*num_encoding_func*3))

    """
    encoding = []

    frequency_band = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_func - 1),
            num_encoding_func,
            dtype=tensor.dtype,
            device=device,
    )

    # Iterate through each frequency and both sin/cos functions to generate the encoded output
    for frequency in frequency_band:
        for func in [torch.sin, torch.cos]:
            sinusoidal_component = func(tensor * frequency)
            encoding.append(sinusoidal_component)

    return torch.cat(encoding, dim=-1)

def perform_integrated_positional_encoding(num_encoding_func: int, device: torch.device, mu_tensor: torch.Tensor, diag_sigma_tensor: torch.tensor=None):
    """
    Computes integrated positional encoding (IPE) for a given tensor representing mean and diagonal covariance.

    Args:
        mu_tensor (torch.Tensor): Tensor representing the mean of the coarse sample points. Shape is (x, y, 3)
        diag_sigma_tensor (torch.Tensor)(optional): Tensor representing the diagonal of the covariance matrix associated with these points. Shape is (x, y, 3)
        num_encoding_func (int): The number of encoding functions to use. Determines the number of sinusoidal functions to encode spatial information.

    Returns:
        torch.Tensor: A tensor representing the integrated positional encoding. 
            Shape is (x, y, num_encoding_func * 3 * 2).
    """
    encoding = []

    # # Represent the frequencies considered in the form of a matrix for the 3 dimensions
    # frequency_matrix =  torch.cat([2**i * torch.eye(3) for i in range(0, num_encoding_func)], dim=-1).to(device)

    # mu_pe = torch.matmul(mu_tensor, frequency_matrix)**2
    # # TODO: check this.  # TODO : Check if cos component helps
    # mu_pe = torch.cat((mu_pe, mu_pe + 0.5 * torch.pi), -1)

    # diag_sigma_pe = torch.matmul(diag_sigma_tensor, frequency_matrix)**2
    # diag_sigma_pe = torch.cat((diag_sigma_pe, diag_sigma_pe), -1)

    # # Using formula derived in the paper for MIP NeRF
    # encoding.append(torch.sin(mu_pe) * torch.exp(-0.5*diag_sigma_pe))

    frequency_vector = torch.Tensor([2 ** i for i in range(0, num_encoding_func)]).to(device)                               
    mu_pe = (mu_tensor[..., None, :] * frequency_vector[:, None]).reshape(list(mu_tensor.shape[:-1]) + [-1])
    mu_pe = torch.cat((mu_pe, mu_pe + 0.5 * torch.pi), -1)

    if diag_sigma_tensor is not None:
        diag_sigma_pe = (diag_sigma_tensor[..., None, :] * frequency_vector[:, None]**2).reshape(list(mu_tensor.shape[:-1]) + [-1])
        diag_sigma_pe = torch.cat((diag_sigma_pe, diag_sigma_pe), -1)
        encoding.append(torch.sin(mu_pe) * torch.exp(-0.5*diag_sigma_pe))
    else:
        encoding.append(torch.sin(mu_pe))

    return torch.cat(encoding, dim=-1)

def sample_pdf(bins, weights, num_samples, sample_randomly):
    """
    Generates samples from a probability distribution defined by bins and weights using inverse transform sampling technique
    (https://en.wikipedia.org/wiki/Inverse_transform_sampling).
    In this context, PDF is of coarse sample points with their likelihood.

    Args:
        bins (torch.Tensor): Tensor representing the bins of the distribution.
        weights (torch.Tensor): Tensor representing the weights associated with each bin.
        num_samples (int): Number of samples to generate.
        sample_randomly (bool): Whether to sample randomly (True) or deterministically (False).

    Returns:
        torch.Tensor: A tensor containing the generated samples.
    """
    
    # Add small number to prevent nans in weights
    weights = weights + 1e-5
    
    # Calculate PDF. Divide by sum to get them between 0 and 1
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