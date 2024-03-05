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