# Libraries
import torch
import torch.nn as nn

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.load_dataset import load_nerf_dataset
from dataset.load_dataset import load_tinynerf_dataset
from models.nerf import NeRF
from utils.run_nerf import run_nerf
from utils.common_utils import mse2psnr
import shutil

from utils.model_utils import load_checkpoint_model
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
    num_coarse_samples_per_ray = 64
    num_fine_samples_per_ray = 64
    include_input_in_posenc = False
    include_input_in_direnc = False
    is_ndc_required = False # set to True only for forward facing scenes
    num_random_rays = 0 # Random Rays Sampling
    use_viewdirs = True
    height, width, focal_length = hwf_list
    focal_length = torch.from_numpy(focal_length).to(device)

    # Define NN model related params
    num_epochs = 1000
    batch_size = 1 # TODO: why not have more images in batch
    chunk_size = 4096 # 16384 # because 4096 for 1.2GB of GPU memory
    validate_every = 20
    save_checkpoint_every = 10000
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
    val_loss_yaxis = []
    val_indices_to_plot = [101] #,167,192]
    

    # Creating directories if not already created
    if os.path.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    
    os.makedirs(os.path.join(LOGDIR, "train", "psnr"), exist_ok=True)
    os.makedirs(os.path.join(LOGDIR, "train", "loss"), exist_ok=True)
    os.makedirs(os.path.join(LOGDIR, "val", "loss"), exist_ok=True)
    os.makedirs(os.path.join(LOGDIR, "val", "psnr"), exist_ok=True)
    os.makedirs(os.path.join(LOGDIR, "models"), exist_ok=True)
    
    for val_idx in val_indices_to_plot:
        name = f"val_{val_idx}"
        os.makedirs(os.path.join(LOGDIR, "val",name, "ground_truth"), exist_ok=True)
        os.makedirs(os.path.join(LOGDIR, "val",name, "coarse_img"), exist_ok=True)
        os.makedirs(os.path.join(LOGDIR, "val",name, "fine_img"), exist_ok=True)
    
    # Saving Validation ground_truth, coarse and fine rendered images 
    val_img_target = images[101].to(device)
    plt.figure()
    plt.imshow(val_img_target.detach().cpu().numpy())
    plt.title(f"Ground truth")
    plt.savefig(os.path.join(LOGDIR, "val", f"val_101", "ground_truth", f"gt.png"))
    plt.clf()

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
        if num_random_rays>0:
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
                os.path.join(LOGDIR, "models","checkpoint" + str(epoch).zfill(5) + ".tar"),
            )
        

        # Evaluate on validation dataset
        if epoch % validate_every == 0:
            # Add values for PSNR graph
            train_loss_yaxis.append(total_loss.item())
            train_psnr_yaxis.append(mse2psnr(total_loss.item()))
            epochs_xaxis.append(epoch)
            print(f"{epoch} Train loss: {total_loss.item()} Train PSNR : {mse2psnr(total_loss.item())}")
            plt.plot(epochs_xaxis, train_psnr_yaxis)
            plt.title(f"Training PSNR Plot {epoch}")
            plt.savefig(os.path.join(LOGDIR, "train", "psnr", f"psnr_{epoch}.png"))
            plt.clf()
            plt.plot(epochs_xaxis, train_loss_yaxis)
            plt.title(f"Training Loss Plot {epoch}")
            plt.savefig(os.path.join(LOGDIR, "train", "loss", f"loss_{epoch}.png"))
            plt.clf()

            val_loss_list = []
            val_psnr_list = []
            with torch.no_grad():
                for val_index in val_indices_to_plot: # TODO : check if all or just randomly one at a time
                    # Run forward pass in eval mode
                    val_img_target = images[val_index].to(device)
                    val_img_target = val_img_target.reshape(-1, 3)
                    val_pose = poses[val_index].to(device)
    
                    rgb_val_coarse, rgb_val_fine, _ = run_nerf(height, width, focal_length, val_pose, use_viewdirs, is_ndc_required,use_white_bkgd,
                                near_thresh, far_thresh, num_coarse_samples_per_ray, num_fine_samples_per_ray,
                                include_input_in_posenc, include_input_in_direnc, num_pos_encoding_functions,
                                num_dir_encoding_functions, model_coarse, model_fine, chunk_size, num_random_rays, mode='eval')
                    # val_img_target = val_img_target[val_selected_ray_indices, :]
    
                    coarse_loss = torch.nn.functional.mse_loss(rgb_val_coarse, val_img_target)
                    fine_loss = torch.nn.functional.mse_loss(rgb_val_fine, val_img_target)
                    total_loss = coarse_loss + fine_loss 
                    val_psnr = mse2psnr(total_loss.item())
    
                    val_loss_list.append(total_loss.item())
                    val_psnr_list.append(val_psnr)
    
                    print(f"{epoch} {val_index} Val loss: {total_loss.item()} Val PSNR: {val_psnr}")
    
                    rgb_val_fine = rgb_val_fine.reshape(height, width, 3)
                    rgb_val_coarse = rgb_val_coarse.reshape(height, width, 3)
    
    
                    plt.imshow(rgb_val_coarse.detach().cpu().numpy())
                    plt.title(f"Coarse RGB {epoch}")
                    plt.savefig(os.path.join(LOGDIR, "val", f"val_{val_index}", "coarse_img", f"coarse_{epoch}.png"))
                    plt.clf()
                    
                    plt.imshow(rgb_val_fine.detach().cpu().numpy())
                    plt.title(f"Fine RGB {epoch}")
                    plt.savefig(os.path.join(LOGDIR, "val", f"val_{val_index}", "fine_img", f"fine_{epoch}.png")) 
                    plt.clf()
                    # # Check if val_index is to be plotted
                    # os.makedirs(os.path.join(LOGDIR, "val", f"val_{val_index}", "psnr",), exist_ok=True)

            # Averaging the Validation loss and PSNR and Saving plots
            val_loss_yaxis.append(sum(val_loss_list) / len(val_indices_to_plot))
            val_psnr_yaxis.append(sum(val_psnr_list) / len(val_indices_to_plot))

            plt.plot(epochs_xaxis, val_psnr_yaxis)
            plt.title(f"Validation PSNR Plot {epoch}")
            plt.savefig(os.path.join(LOGDIR, "val", "psnr", f"psnr_{epoch}.png"))
            plt.clf()

            plt.plot(epochs_xaxis, val_loss_yaxis)
            plt.title(f"Validation Loss Plot {epoch}")
            plt.savefig(os.path.join(LOGDIR, "val", "loss", f"loss_{epoch}.png"))
            plt.clf()
                        
    checkpoint_dict = {
            'epoch': epoch, 
            'model_coarse_state_dict': model_coarse.state_dict(), 
            "model_fine_state_dict": model_fine.state_dict(), 
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss.item()
    }
    torch.save(
                checkpoint_dict,
                os.path.join(LOGDIR, "models","final_model_" + str(epoch) + ".tar"),
    )
    
    # with torch.no_grad():
    #     op, model_coarse, model_fine = load_checkpoint_model("/scratch/sravindh/project_nerf/logs/models/final_model_2.tar", optimizer, model_coarse, model_fine)
    #     c, f, _ = run_nerf(height, width, focal_length, val_pose, use_viewdirs, is_ndc_required,use_white_bkgd,
    #                             near_thresh, far_thresh, num_coarse_samples_per_ray, num_fine_samples_per_ray,
    #                             include_input_in_posenc, include_input_in_direnc, num_pos_encoding_functions,
    #                             num_dir_encoding_functions, model_coarse, model_fine, chunk_size, num_random_rays, mode='eval')
    #     print(c.reshape(h,3).shape)
    #     print("Eval")
            
def main():


    # Load dataset
    # images, poses, hwf_list, train_indices, val_indices, test_indices, \
    # sph_test_poses, near_thresh, far_thresh= load_nerf_dataset(DATASET_TYPE, DATASET_DIR)
    # images, poses, hwf_list, train_indices, val_indices, near_thresh, far_thresh= load_tinynerf_dataset()
    images = torch.from_numpy(images)
    poses = torch.from_numpy(poses)

    # Call the train or inference function # TODO: add appropriate classes
    # train_nerf(images, poses, hwf_list, train_indices, val_indices, near_thresh, far_thresh)    

    # TODO: add inference        

if __name__ == "__main__":
    main()