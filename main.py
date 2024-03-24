import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import imageio

from utils.common_utils import * 
from dataset.load_dataset import load_nerf_dataset
from models.nerf import NeRF
from utils.run_nerf import run_nerf
from utils.model_utils import load_model_checkpoint

# TODO: val set
# TODO: height, width crct ??
# TODO: decaying learning rate 
# TODO: check param values

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def eval_nerf(cfg, poses: torch.Tensor, hwf_list: list, 
              near_thresh: float, far_thresh: float):
    #TODO: Add rendering time, log it if necessary
    height, width, focal_length = hwf_list
    poses = torch.from_numpy(poses)
    poses = poses.to(cfg.device)


    # Define the models and the optimizer
    model_coarse = NeRF(cfg.model.num_pos_encoding_func, cfg.model.num_dir_encoding_func, cfg.model.use_viewdirs).to(cfg.device)
    model_fine = NeRF(cfg.model.num_pos_encoding_func, cfg.model.num_dir_encoding_func, cfg.model.use_viewdirs).to(cfg.device)
    optimizer = torch.optim.Adam(list(model_coarse.parameters()) + list(model_fine.parameters()), lr=float(cfg.train.lr))

    # Check if there is a pretrained checkpoint that is available
    if cfg.train.checkpoint_path:
        print(f"Loading pretrained model from checkpoint path: {cfg.train.checkpoint_path}")
        _, optimizer, model_coarse, model_fine = load_model_checkpoint(cfg.train.checkpoint_path, optimizer, model_coarse, model_fine)
    
    if not os.path.exists(os.path.join(cfg.result.logdir, 'test')):
        os.mkdir(os.path.join(cfg.result.logdir, 'test'))
    rgb_list = []
    with torch.no_grad():
        for i in tqdm(range(poses.shape[0])):
            _, rgb_test_fine, _ = run_nerf(height, width, focal_length, poses[i],
                near_thresh, far_thresh, model_coarse, model_fine, cfg, 0, mode='eval')
            rgb_test_fine = rgb_test_fine.reshape(height, width, 3)
            rgb_list.append(rgb_test_fine)
            imageio.imwrite(os.path.join(cfg.result.logdir, 'test', '{:03d}.png'.format(i)), cast_tensor_to_image(rgb_test_fine))

    # Save Video for 360 rendering
    if cfg.result.is_spherical_rendering:
        rgb_list = torch.cat(rgb_list, dim=-1)
        imageio.mimwrite(os.path.join(cfg.result.logdir, 'test', 'test_video.mp4'), cast_tensor_to_image(rgb_list), fps=30, quality=8)
    print(f"Done Rendering")

def train_nerf(cfg, images:torch.Tensor, poses: torch.Tensor, hwf_list: list, 
               train_indices: list, near_thresh: float, far_thresh: float):
    """
    Train a NeRF model using the specified images, camera poses, and other parameters.

    Args:
        cfg: dictionary like object containing user config
        images (torch.Tensor): Tensor containing input images. Shape (num_images, height, width, 4).
        poses (torch.Tensor): Tensor containing camera poses corresponding to each image. Shape (num_images, 4, 4).
        hwf_list (list): List containing the height, width and focal length.
        train_indices (list): List of indices indicating the images used for training.
        near_thresh (float): Near threshold for ray termination.
        far_thresh (float): Far threshold for ray termination.

    Returns:
        None
    """
    height, width, focal_length = hwf_list

    # Initialize summary writer
    writer = SummaryWriter(log_dir = os.path.join(cfg.result.logdir, "tf_writer"))
    os.makedirs(os.path.join(cfg.result.logdir, "model_checkpoints"), exist_ok=True)
    # shutil.mkdir(os.path.join(cfg.result.logdir, "model_checkpoints"), parents=True, exist_ok=True)

    # Define validation data
    val_index = 111
    val_target_img = images[val_index].to(cfg.device)
    writer.add_image("valimages/ground_truth", cast_tensor_to_image(val_target_img), 0) 

    # Define the models and the optimizer
    model_coarse = NeRF(cfg.model.num_pos_encoding_func, cfg.model.num_dir_encoding_func, cfg.model.use_viewdirs).to(cfg.device)
    model_fine = NeRF(cfg.model.num_pos_encoding_func, cfg.model.num_dir_encoding_func, cfg.model.use_viewdirs).to(cfg.device)
    optimizer = torch.optim.Adam(list(model_coarse.parameters()) + list(model_fine.parameters()), lr=float(cfg.train.lr))

    # Check if there is a pretrained checkpoint that is available
    if cfg.train.checkpoint_path:
        cfg.result.logger.info(f"Loading pretrained model from checkpoint path: {cfg.train.checkpoint_path}")
        start_epoch, optimizer, model_coarse, model_fine = load_model_checkpoint(cfg.train.checkpoint_path, optimizer, model_coarse, model_fine)
    else:
        start_epoch = 0
    # Iterate through epochs
    cfg.result.logger.info(f"Initiating model training.")
    for epoch in tqdm(range(start_epoch, start_epoch+cfg.train.num_epochs)):

        # Pick one random sample for training
        index = np.random.choice(train_indices) # TODO: check if it is without replacement
        target_img = images[index].to(cfg.device)
        training_campose = poses[index].to(cfg.device)

        # Run forward pass
        rgb_coarse, rgb_fine, selected_ray_coors = run_nerf(height, width, focal_length, training_campose,
                near_thresh, far_thresh, model_coarse, model_fine, cfg, epoch, mode='train')
        if cfg.model.num_selected_rays > 0:
            target_img = target_img[selected_ray_coors[:, 0], selected_ray_coors[:, 1]]
        
        # Compute loss
        coarse_loss = torch.nn.functional.mse_loss(rgb_coarse, target_img)
        fine_loss = torch.nn.functional.mse_loss(rgb_fine, target_img)
        total_loss = coarse_loss + fine_loss

        # Backpropagate
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update the learning rate
        decay_rate = 0.1
        decay_steps = cfg.train.lr_decay * 1000
        cfg.train.lr = float(cfg.train.lr) * (decay_rate ** (epoch / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.train.lr

        # Save model checkpoint
        if (epoch%cfg.train.save_checkpoint_for_every == 0) or (epoch == cfg.train.num_epochs):
            checkpoint_dict = {
                'epoch': epoch, 
                'model_coarse_state_dict': model_coarse.state_dict(), 
                "model_fine_state_dict": model_fine.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss.item()
            }
            torch.save(
                checkpoint_dict,
                os.path.join(cfg.result.logdir, "model_checkpoints", "checkpoint"+str(epoch).zfill(6)+".tar"),
            )
        
        # Save training loss and psnr values to writer
        writer.add_scalar('train/loss', total_loss.item(), epoch)
        writer.add_scalar('train/psnr', convert_mse_to_psnr(total_loss.item()), epoch)

        # Evaluate on validation data
        if epoch % cfg.train.validate_every == 0:

            with torch.no_grad():
                val_target_img = val_target_img.reshape(-1, 3)
                val_pose = poses[val_index].to(cfg.device)
                rgb_val_coarse, rgb_val_fine, _ = run_nerf(height, width, focal_length, val_pose,
                    near_thresh, far_thresh, model_coarse, model_fine, cfg, 0, mode='eval')
    
                val_coarse_loss = torch.nn.functional.mse_loss(rgb_val_coarse, val_target_img)
                val_fine_loss = torch.nn.functional.mse_loss(rgb_val_fine, val_target_img)
                val_total_loss = val_coarse_loss + val_fine_loss 

                writer.add_scalar('val/loss', val_total_loss.item(), epoch)
                writer.add_scalar('val/psnr', convert_mse_to_psnr(val_total_loss.item()), epoch)
    
                rgb_val_fine = rgb_val_fine.reshape(height, width, 3)
                rgb_val_coarse = rgb_val_coarse.reshape(height, width, 3)
                writer.add_image("valimages/coarse", cast_tensor_to_image(rgb_val_coarse), epoch)
                writer.add_image("valimages/fine", cast_tensor_to_image(rgb_val_fine), epoch)
            
def main():
    # Read user configurable settings from config file
    config_file = "./configs/nerf_lego.yaml"
    cfg = read_config(config_file)

    # Define device to be used
    cfg.device = torch.device(cfg.device) 
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    # Load dataset
    images, poses, hwf_list, train_indices, val_indices, test_indices, \
    sph_test_poses, near_thresh, far_thresh = load_nerf_dataset(cfg.dataset.type, cfg.dataset.dir)
    images = torch.from_numpy(images)
    poses = torch.from_numpy(poses)

    # Create result directory if not already created
    if not os.path.exists(cfg.result.logdir):
        os.mkdir(cfg.result.logdir)
    else:
        shutil.rmtree(cfg.result.logdir)
    
    # Create a log file
    # log_file_path = os.path.join(cfg.result.logdir, "logfile.txt")
    cfg.result.logger = logging.getLogger()
    # cfg.result.logger.setLevel(logging.INFO)
    # file_handler = logging.Filehandler(log_file_path)
    # file_handler.setLevel(logging.INFO)
    # cfg.result.logger.addHandler(file_handler)

    # TODO: print config in log file

    # Call the train or inference function
    train_nerf(cfg, images, poses, hwf_list, train_indices, near_thresh, far_thresh) 
    
    if cfg.result.is_spherical_rendering:
        eval_nerf(cfg, sph_test_poses, hwf_list, near_thresh, far_thresh) 
    else:
        test_poses = poses[test_indices]
        eval_nerf(cfg, test_poses, hwf_list, near_thresh, far_thresh) 
    
if __name__ == "__main__":
    main()