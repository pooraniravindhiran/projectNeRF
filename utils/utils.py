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

def load_checkpoint_model(checkpoint_path:str, optimizer, model_coarse, model_fine):
    checkpoint = torch.load(checkpoint_path)
    start_iter = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load model
    model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])
    model_fine.load_state_dict(checkpoint['model_fine_state_dict'])

    return start_iter, optimizer, model_coarse, model_fine