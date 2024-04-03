import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse
import imageio
import time

from utils.common_utils import * 
from dataset.load_dataset import load_nerf_dataset
from models.nerf import NeRF
from utils.run_mipnerf import run_mipnerf
from utils.model_utils import load_model_checkpoint

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def eval_mipnerf(cfg, poses: torch.Tensor, hwf_list: list, 
              near_thresh: float, far_thresh: float, 
              ground_truth=None):
    height, width, focal_length = hwf_list
    if not isinstance(poses, torch.Tensor):
        poses = torch.from_numpy(poses)
    poses = poses.to(cfg.device)

    # Define the models and the optimizer
    model = NeRF(cfg.model.num_pos_encoding_func, cfg.model.num_dir_encoding_func, cfg.model.use_viewdirs).to(cfg.device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=float(cfg.train.lr))

    # Check if there is a pretrained checkpoint that is available
    if cfg.train.checkpoint_path:
        _, optimizer, model, _ = load_model_checkpoint(cfg, optimizer, model, None)
        cfg.result.logger.info(f"Loaded pretrained model from checkpoint path: {cfg.train.checkpoint_path}.")

    # Create output dir
    if not os.path.exists(os.path.join(cfg.result.logdir, 'inference')):
        os.mkdir(os.path.join(cfg.result.logdir, 'inference'))

    # Run inference for all test poses
    rgb_list = []
    avg_loss = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    with torch.no_grad():
        start_time = time.time()
        for i in tqdm(range(poses.shape[0])):
            rgb_test_coarse, rgb_test_fine, _ = run_mipnerf(height, width, focal_length, poses[i],
                near_thresh, far_thresh, model, cfg, 0, mode='eval')
            if ground_truth is not None:
                target_image = ground_truth[i].view(-1, 3)
                coarse_loss = torch.nn.functional.mse_loss(rgb_test_coarse, target_image)
                fine_loss = torch.nn.functional.mse_loss(rgb_test_fine, target_image)
                total_loss = (coarse_loss*0.1)+fine_loss
                avg_loss += total_loss.item()
                avg_psnr += convert_mse_to_psnr(total_loss.item())
                avg_ssim += compute_ssim_score(rgb_test_fine.reshape(height, width, 3).detach().cpu().numpy(), \
                                               target_image.reshape(height, width, 3).detach().cpu().numpy())
            rgb_test_fine = rgb_test_fine.reshape(height, width, 3)
            rgb_test_fine = np.moveaxis(cast_tensor_to_image(rgb_test_fine), 0, -1)
            rgb_list.append(rgb_test_fine)

            imageio.imwrite(os.path.join(cfg.result.logdir, 'inference', '{:03d}.png'.format(i)), rgb_test_fine)
        end_time = time.time()

    # Log time info
    cfg.result.logger.info(f"Total time to render {poses.shape[0]} images: {end_time-start_time}s.")
    if ground_truth is not None:
        cfg.result.logger.info(f"Average loss: {avg_loss/poses.shape[0]}\tavg psnr: {avg_psnr/poses.shape[0]} \tavg ssim: {avg_ssim/ poses.shape[0]}")

    # Save video for 360 rendering
    if ground_truth is None: 
        rgb_list = np.stack(rgb_list, dim=-1)
        imageio.mimwrite(os.path.join(cfg.result.logdir, 'inference', 'test_video.mp4'), rgb_list, fps=30, quality=8)

    cfg.result.logger.info(f"Test images are rendered.")

def train_mipnerf(cfg, images:torch.Tensor, poses: torch.Tensor, hwf_list: list, 
               train_indices: list, near_thresh: float, far_thresh: float):
    """
    Train a Mip NeRF model using the specified images, camera poses, and other parameters.

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
    model = NeRF(cfg.model.num_pos_encoding_func, cfg.model.num_dir_encoding_func, cfg.model.use_viewdirs).to(cfg.device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=float(cfg.train.lr))
    
    # Check if there is a pretrained checkpoint that is available
    if cfg.train.checkpoint_path:
        start_epoch, optimizer, model, _ = load_model_checkpoint(cfg.train.checkpoint_path, optimizer, model, None)
        cfg.result.logger.info(f"Loaded pretrained model from checkpoint path: {cfg.train.checkpoint_path}.")
    else:
        start_epoch = 0

    # Iterate through epochs
    cfg.result.logger.info(f"Initiating model training.")
    training_start_time = time.time()
    for epoch in tqdm(range(start_epoch, start_epoch+cfg.train.num_epochs+1)):

        # Pick one random sample for training
        index = np.random.choice(train_indices) # TODO: check if it is without replacement
        target_img = images[index].to(cfg.device)
        training_campose = poses[index].to(cfg.device)

        # Run forward pass
        rgb_coarse, rgb_fine, selected_ray_coors = run_mipnerf(height, width, focal_length, training_campose,
                near_thresh, far_thresh, model, cfg, epoch, mode='train')
        if cfg.model.num_selected_rays > 0:
            target_img = target_img[selected_ray_coors[:, 0], selected_ray_coors[:, 1]]
        # Compute loss
        coarse_loss = torch.nn.functional.mse_loss(rgb_coarse, target_img)
        fine_loss = torch.nn.functional.mse_loss(rgb_fine, target_img)
        total_loss = (coarse_loss * 0.1) + fine_loss

        # Compute ssim
        # ssim_score = compute_ssim_score(rgb_fine.detach().cpu().numpy(), target_img.detach().cpu().numpy())

        # Backpropagate
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update the learning rate
        decay_rate = 0.1
        decay_steps = cfg.train.lr_decay * 1000
        new_lr = float(cfg.train.lr) * (decay_rate ** (epoch / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        # Save model checkpoint
        if (epoch%cfg.train.save_checkpoint_for_every == 0 and epoch != 0) or (epoch == start_epoch+cfg.train.num_epochs):
            checkpoint_dict = {
                'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss.item()
            }
            torch.save(
                checkpoint_dict,
                os.path.join(cfg.result.logdir, "model_checkpoints", "checkpoint"+str(epoch).zfill(6)+".tar"),
            )
        
        # Save training loss and psnr values to writer
        writer.add_scalar('train/loss', total_loss.item(), epoch)
        # writer.add_scalar('train/ssim', ssim_score, epoch)
        writer.add_scalar('train/psnr', convert_mse_to_psnr(total_loss.item()), epoch)
        writer.add_scalar('lr', new_lr ,epoch)

        # Evaluate on validation data
        if (epoch % cfg.train.validate_every == 0 and epoch != 0) or (epoch == start_epoch+cfg.train.num_epochs):

            with torch.no_grad():
                
                val_target_img = val_target_img.reshape(-1, 3)
                val_pose = poses[val_index].to(cfg.device)
                rgb_val_coarse, rgb_val_fine, _ = run_mipnerf(height, width, focal_length, val_pose,
                    near_thresh, far_thresh, model, cfg, 0, mode='eval')
    
                val_coarse_loss = torch.nn.functional.mse_loss(rgb_val_coarse, val_target_img)
                val_fine_loss = torch.nn.functional.mse_loss(rgb_val_fine, val_target_img)
                val_total_loss = (val_coarse_loss*0.1) + val_fine_loss 

                writer.add_scalar('val/loss', val_total_loss.item(), epoch)
                writer.add_scalar('val/psnr', convert_mse_to_psnr(val_total_loss.item()), epoch)

    
                rgb_val_fine = rgb_val_fine.reshape(height, width, 3)
                rgb_val_coarse = rgb_val_coarse.reshape(height, width, 3)
                rgb_val_coarse = cast_tensor_to_image(rgb_val_coarse)
                rgb_val_fine =  cast_tensor_to_image(rgb_val_fine)

                val_ssim = compute_ssim_score(rgb_val_fine, cast_tensor_to_image(val_target_img.reshape(height, width, 3)))
                
                writer.add_image("valimages/coarse", rgb_val_coarse, epoch)
                writer.add_image("valimages/fine", rgb_val_fine, epoch)
                writer.add_scalar('val/ssim', val_ssim, epoch)
                
                # Add metrics to the logger
                cfg.result.logger.info(f"Train Epoch: {epoch}\t loss: {total_loss.item()}\tpsnr: {convert_mse_to_psnr(total_loss.item())}")
                cfg.result.logger.info(f"Val Epoch: {epoch}\t loss: {val_total_loss.item()}\tpsnr: {convert_mse_to_psnr(val_total_loss.item())}\tssim: {val_ssim}")
    
    training_end_time = time.time()
    time_difference = training_end_time - training_start_time
    hours = int(time_difference // 3600)
    minutes = int((time_difference % 3600) // 60)
    seconds = int(time_difference % 60)
    writer.flush()
    writer.close()
    cfg.result.logger.info(f"Completed model training successfully.")
    cfg.result.logger.info("Training time : {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds))

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help="For training model, type train. For inference, type eval.")
    parser.add_argument('--logdir', type=str, help="Provide the path to save the results and logs.")
    parser.add_argument('--model_path', type=str, help="Provide the path to the model saved if any.", default=None)
    parser.add_argument('--is_spherical', type=bool, help="Indicate if you want spherical poses or test poses for evaluation.", default=False)
    args = parser.parse_args()

    # Read user configurable settings from config file
    config_file = "./configs/mipnerf_lego.yaml"
    cfg = read_config(config_file)

    # Define device to be used
    cfg.device = torch.device(cfg.device) 
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    
    # Create result directory if not already created
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    else: 
        if args.mode == "train":
            shutil.rmtree(os.path.dirname(args.logdir))
            os.mkdir(args.logdir)
    cfg.result.logdir = args.logdir
    cfg.train.checkpoint_path = args.model_path
    
    # Configure logging
    log_file_path = os.path.join(args.logdir, "logfile.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w'  # Set file mode to 'w' to overwrite existing log file
    )
    # Create a console handler and set the level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create a formatter and attach it to the console handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)
    cfg.result.logger = logging
    cfg.result.logger.info(f"Created log file: {log_file_path}.")

    # Add config to the log file
    cfg.result.logger.info(f"\nBelow is the user config.")
    for section, options in cfg.items():
        if isinstance(options, dict):
            cfg.result.logger.info(f"Tags: {section}")
            for key, value in options.items():
                cfg.result.logger.info(f"{key}: {value}")
    cfg.result.logger.info("\n")

    # Load dataset
    images, poses, hwf_list, train_indices, val_indices, test_indices, \
    sph_test_poses, near_thresh, far_thresh = load_nerf_dataset(cfg)
    images = torch.from_numpy(images)
    poses = torch.from_numpy(poses)
    cfg.result.logger.info(f"Loaded dataset from {cfg.dataset.dir} successfully.")
    
    # Call the train or inference function
    if args.mode == "train":
        train_mipnerf(cfg, images, poses, hwf_list, train_indices, near_thresh, far_thresh) 
    else:
        if args.is_spherical:
            eval_mipnerf(cfg, sph_test_poses, hwf_list, near_thresh, far_thresh) 
        else:
            test_poses = poses[test_indices]
            eval_mipnerf(cfg, test_poses, hwf_list, near_thresh, far_thresh, images[test_indices]) 
    
if __name__ == "__main__":
    main()