# Libraries
import torch
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.load_dataset import load_nerf_dataset
from models.nerf import NeRF
from utils.run_nerf import run_nerf
from utils.common_utils import mse2psnr

# TODO: val set
# TODO: height, width crct ??
# TODO: decaying learning rate 
# TODO: check param values

# Setting random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting global variables
DATASET_DIR = '/scratch/sravindh/nerf_dataset/lego/'
DATASET_TYPE = 'lego'
LOGDIR = '/scratch/sravindh/project_nerf/logs/'

def train_nerf(images, poses, hwf_list, train_indices, val_indices, near_thresh, far_thresh):
    # Define ray related params
    num_pos_encoding_functions = 6
    num_dir_encoding_functions = 6
    num_coarse_samples_per_ray = 32
    num_fine_samples_per_ray = 32
    include_input_in_posenc = False
    include_input_in_direnc = False
    is_ndc_required = False # set to True only for forward facing scenes
    num_random_rays = 1024 # Random Rays Sampling
    use_viewdirs = True
    height, width, focal_length = hwf_list

    # Define NN model related params
    num_epochs = 1
    batch_size = 1 # TODO: why not have more images in batch
    chunk_size = 4096   #16384 # because 4096 for 1.2GB of GPU memory
    validate_every = 1
    save_checkpoint_every = 1
    checkpoint_model = ''
    lr = 5e-3
    # lrate_decay = 250
    # update_lr_every = 0

    # Define data related params
    use_white_bkgd = False # for Lego synthetic data # TODO: check this

    # Define plotting related params
    epochs_xaxis = []
    train_psnr_yaxis = []
    val_psnr_yaxis = []
    train_loss_yaxis = []
    val_loss_y_axis = []
    val_indices_to_plot = [111,167,192]

    # Define the models and the optimizer
    model_coarse = NeRF(num_pos_encoding_functions, num_dir_encoding_functions, use_viewdirs).to(device)
    model_fine = NeRF(num_pos_encoding_functions, num_dir_encoding_functions, use_viewdirs).to(device)
    optimizer = torch.optim.Adam(list(model_coarse.parameters()) + list(model_fine.parameters()), lr=lr)

    # Iterate through epochs
    print("Training has begun.\n")
    for epoch in tqdm(range(num_epochs)):

        # Pick one random sample for training
        index = np.random.choice(train_indices) # TODO: check if it is without replacement
        target_img = images[index].to(device)
        target_img = target_img.reshape(-1, 3)
        training_campose = poses[index].to(device)

        # Run forward pass
        rgb_coarse, rgb_fine, selected_ray_indices = run_nerf(height, width, focal_length, training_campose, use_viewdirs, is_ndc_required, use_white_bkgd,
                near_thresh, far_thresh, num_coarse_samples_per_ray, num_fine_samples_per_ray,
                include_input_in_posenc, include_input_in_direnc, num_pos_encoding_functions,
                num_dir_encoding_functions, model_coarse, model_fine, chunk_size, num_random_rays, mode='train')
        target_img = target_img[selected_ray_indices, :]
        
        # Compute loss
        coarse_loss = torch.nn.functional.mse_loss(rgb_coarse, target_img)
        fine_loss = torch.nn.functional.mse_loss(rgb_fine, target_img)
        total_loss = coarse_loss + fine_loss #TODO - why summing it ?

        # Backpropagate
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # # Update the learning rate
        # if update_lr_every != 0:
        #     if epoch % update_lr_every == 0:
        #         decay_rate = 0.1
        #         decay_steps = lrate_decay * 1000
        #         new_lr = lr * (decay_rate ** (epoch / decay_steps))

        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = new_lr

        # Save model at checkpoints
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
        
        # Add values for PSNR graph
        train_loss_yaxis.append(total_loss.item())
        train_psnr_yaxis.append(mse2psnr(total_loss.item()))
        epochs_xaxis.append(epoch)

        # Evaluate on validation dataset
        # if epoch % validate_every == 0:
            # for val_index in val_indices: # TODO : check if all or just randomly one at a time
                # Run forward pass in eval mode

                # Check if val_index is to be plotted
            
            # print(f"{epoch} Train loss: {total_loss.item()} Train PSNR : {mse2psnr(total_loss.item())}")
            # plt.plot(epochs_x, psnr_y)
            # plt.title(f"PSNR Graph {epoch}")
            # plt.savefig(f"{LOGDIR}psnr/psnr_{epoch}.png")

            #         for val_image_idx in val_image_indices:

            #             val_img_target = images[val_image_idx].to(device)
            #             val_img_target = val_img_target.reshape(-1, 3)
            #             val_pose = poses[val_image_idx].to(device)

            #             rgb_val_coarse, rgb_val_fine, val_random_indices = run_Nerf(height, width, focal_length, val_pose, use_viewdirs, is_ndc_required,use_white_bkgd,
            #                 near_thresh, far_thresh, num_coarse_samples_per_ray, num_fine_samples_per_ray,
            #                 include_input_in_posenc, include_input_in_direnc, num_pos_encoding_functions,
            #                 num_dir_encoding_functions, model_coarse, model_fine, chunk_size, num_random_rays, mode='eval')
                        
            #             val_img_target = val_img_target[val_random_indices, :]
            #             coarse_loss = torch.nn.functional.mse_loss(rgb_val_coarse, val_img_target)
            #             fine_loss = torch.nn.functional.mse_loss(rgb_val_fine, val_img_target)
            #             total_loss = coarse_loss + fine_loss 

            #             rgb_val_fine = rgb_val_fine.reshape(height, width, 3)
            #             rgb_val_coarse = rgb_val_coarse.reshape(height, width, 3)
            #             print(f"{epoch} {val_image_idx} Val loss: {total_loss.item()} Val PSNR: {mse2psnr(total_loss.item())}")
                        
                        
            #             plt.imshow(rgb_val_fine.detach().cpu().numpy())
            #             plt.title(f"Fine {epoch}")
            #             plt.savefig(f"{LOGDIR}val/val_{val_image_idx}/fine/fine{epoch}.png")

            #             plt.imshow(val_img_target.detach().cpu().numpy())
            #             plt.title(f"Ground truth")
            #             plt.savefig(f"{LOGDIR}val/val_{val_image_idx}/gt/gt{epoch}.png")

            #             plt.imshow(rgb_val_coarse.detach().cpu().numpy())
            #             plt.title(f"Coarse {epoch}")
            #             plt.savefig(f"{LOGDIR}val/val_{val_image_idx}/coarse/coarse{epoch}.png")
            #             # plt.show()  
            
def main():
    # Load dataset
    images, poses, hwf_list, train_indices, val_indices, test_indices, \
    sph_test_poses, near_thresh, far_thresh= load_nerf_dataset(DATASET_TYPE, DATASET_DIR)
    images = torch.from_numpy(images)
    poses = torch.from_numpy(poses)

    # Call the train or inference function # TODO: add appropriate classes
    train_nerf(images, poses, hwf_list, train_indices, val_indices, near_thresh, far_thresh)    

    # TODO: add inference        

if __name__ == "__main__":
    main()