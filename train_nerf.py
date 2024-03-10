# Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.common_utils import mse2psnr
from dataset.load_dataset import load_nerf_dataset
from models.nerf import NeRF
from utils.run_nerf import run_Nerf

# [COMPLETED] TODO: device 
# [COMPLETED] TODO: test, val images
#  TODO: lego - half_resolution, white backgnd, near, far
# [COMPLETED] TODO: height, width crct ??
# TODO decaying learning rate 

# Reproducibility - Setting Random Seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATASET_DIR = '...'
DATASET_TYPE = 'lego'
LOGDIR = '...'

# TODO check param values
use_saved_model = False
checkpoint_model = ''
save_checkpoint_every = 1000
lr = 5e-3
lrate_decay = 250

# Loading Dataset - LLFF and Lego 
near_thresh, far_thresh, hwf, \
images, poses, render_poses, i_train, i_val, i_test = load_nerf_dataset(DATASET_TYPE, DATASET_DIR)
    
# Camera intrinsics
height, width, focal_length = hwf
height, width = int(height), int(width)
hwf = [height, width, focal_length]
images = torch.from_numpy(images)
poses = torch.from_numpy(poses)


# Training module

num_pos_encoding_functions = 6
num_dir_encoding_functions = 6
num_coarse_samples_per_ray = 64
num_fine_samples_per_ray = 64
include_input_in_posenc = False
include_input_in_direnc = False
is_ndc_required = False # set to True only for forward facing scenes
use_white_bkgd = False # for Lego synthetic data
use_viewdirs = True
update_lr_every = 5000
num_random_rays = 1024 # Random Rays Sampling

num_epochs = 20000
batch_size = 1 # TODO: why not have more images in batch
chunk_size = 16384 # because 4096 for 1.2GB of GPU memory
validate_every = 500

model_coarse = NeRF(num_pos_encoding_functions, num_dir_encoding_functions, use_viewdirs).to(device)
model_fine = NeRF(num_pos_encoding_functions, num_dir_encoding_functions, use_viewdirs).to(device)
optimizer = torch.optim.Adam(list(model_coarse.parameters()) + list(model_fine.parameters()), lr=lr)
print("================================= TRAINING =============================")
print()
for epoch in tqdm(range(num_epochs)):

    # Pick one random sample for training
    index = np.random.choice(i_train) # TODO: check if it is without replacement
    target_img = images[index].to(device)
    target_img = target_img.reshape(-1, 3)
    training_campose = poses[index].to(device)

    # Call NeRF
    rgb_coarse, rgb_fine = run_Nerf(height, width, focal_length, training_campose, use_viewdirs, is_ndc_required, use_white_bkgd,
            near_thresh, far_thresh, num_coarse_samples_per_ray, num_fine_samples_per_ray,
            include_input_in_posenc, include_input_in_direnc, num_pos_encoding_functions,
            num_dir_encoding_functions, model_coarse, model_fine, chunk_size, num_random_rays, mode='train')

    # Compute total loss - coarse + fine
    coarse_loss = torch.nn.functional.mse_loss(rgb_coarse, target_img)
    fine_loss = torch.nn.functional.mse_loss(rgb_fine, target_img)
    total_loss = coarse_loss + fine_loss #TODO - why summing it ?

    # Backpropagate
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Update the learning rate
    if epoch % update_lr_every == 0:
        decay_rate = 0.1
        decay_steps = lrate_decay * 1000
        new_lr = lr * (decay_rate ** (epoch / decay_steps))

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    # Save model - checkpoints
    if epoch % save_checkpoint_every == 0:
        checkpoint_dict = {
        'epoch': epoch, 
        'model_coarse_state_dict': model_coarse.state_dict(), 
        "model_fine_state_dict": model_fine.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": total_loss.item()
        }

        torch.save(
            checkpoint_dict,
            os.path.join(LOGDIR, "checkpoint" + str(epoch).zfill(5) + ".tar"),
        )

    # Evaluation for every few epochs on test pose
    # Validation
    if epoch % validate_every == 0:
        print(f"{epoch} Train loss: {total_loss.item()} Train PSNR : {mse2psnr(total_loss.item())}")
        val_image_idx = np.random.choice(i_val) # i_val[0]
        val_img_target = images[val_image_idx].to(device)
        val_pose = poses[i_val]
        rgb_val_coarse, rgb_val_fine = run_Nerf(height, width, focal_length, val_pose, use_viewdirs, is_ndc_required,use_white_bkgd,
            near_thresh, far_thresh, num_coarse_samples_per_ray, num_fine_samples_per_ray,
            include_input_in_posenc, include_input_in_direnc, num_pos_encoding_functions,
            num_dir_encoding_functions, model_coarse, model_fine, chunk_size, num_random_rays, mode='eval')
        
        coarse_loss = torch.nn.functional.mse_loss(rgb_coarse, target_img)
        fine_loss = torch.nn.functional.mse_loss(rgb_fine, target_img)
        total_loss = coarse_loss + fine_loss 

        fig = plt.figure()
        rgb_val_fine = rgb_val_fine.reshape(height, width, 3)
        rgb_val_coarse = rgb_val_coarse.reshape(height, width, 3)
        print(f"{epoch} Val loss: {total_loss.item()} Val PSNR: {mse2psnr(total_loss.item())}")
        plt.imshow(rgb_val_fine.detach().cpu().numpy())
        plt.title(f"Iteration {epoch}")
        plt.savefig(f"{LOGDIR}val/val_{val_image_idx}_{epoch}.png")

        plt.figure(figsize=(25, 4))
        plt.subplot(131)
        plt.imshow(val_img_target.detach().cpu().numpy())
        plt.title(f"Ground truth")
        plt.subplot(132)
        plt.imshow(rgb_val_coarse.detach().cpu().numpy())
        plt.title("Coarse")
        plt.subplot(133)
        plt.imshow(rgb_val_fine.detach().cpu().numpy())
        plt.title("Fine")
        plt.savefig(f"{LOGDIR}val_comp/val_{val_image_idx}_{epoch}.png")

        
        # plt.show()