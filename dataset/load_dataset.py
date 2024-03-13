import torch

import numpy as np

from dataset.load_blender import load_blender_data
from dataset.load_llff import load_llff_data

# TODO: log statements
# TODO: add function comments
# TODO: check imports and redefine them if needed
# TODO: check PEP8 compatibility

def load_nerf_dataset(DATASET_TYPE, DATASET_DIR):
    if DATASET_TYPE == 'llff':
        # Define tunable attributes for dataset creation
        every_ith_index_for_test = 8

        # Define near and far plane depth values
        near_thresh = 0.0
        far_thresh = 1.0

        # Call function to create the dataset
        images, poses, _, sph_test_poses, test_indices = load_llff_data(DATASET_DIR, factor=8,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=False)
        hwf_list = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        if not isinstance(test_indices, list):
            test_indices = [test_indices]
        if every_ith_index_for_test > 0:
            test_indices = np.arange(images.shape[0])[::every_ith_index_for_test]
        val_indices = test_indices
        train_indices = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in test_indices and i not in val_indices)])   
        print(f'Loaded llff dataset from {DATASET_DIR}.\nShape of images: {images.shape}.')
        
        return images, poses, hwf_list, train_indices, val_indices, test_indices, sph_test_poses, near_thresh, far_thresh
    
    elif DATASET_TYPE == 'lego':
        # Define near and far plane depth values
        near_thresh = 2.0
        far_thresh = 6.0

        # Define tunable attributes for dataset creation
        half_img_resolution = True
        num_of_test_images_per_set = 1
        use_white_background = False
        
        # Call function to create the dataset
        images, poses, hwf_list, indices_tuple, sph_test_poses = load_blender_data(DATASET_DIR, half_img_resolution, num_of_test_images_per_set)
        print(f'Loaded lego dataset from {DATASET_DIR}.\nShape of images: {images.shape}.')
        
        # Get indices for data split
        train_indices, val_indices, test_indices = indices_tuple
        
        # Manipulate the background
        if use_white_background:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
        
        return images, poses, hwf_list, train_indices, val_indices, test_indices, sph_test_poses, near_thresh, far_thresh
    
    else:
        print('Unknown dataset type. Only lego and llff supported now.')

def load_tinynerf_dataset():
    # Define near and far plane depth values
    near_thresh = 2.0
    far_thresh = 6.0

    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal_length = data['focal']
    height, width = images.shape[1:3]
    hwf_list = [height, width, focal_length]

    # Get indices for data split
    val_indices = [101]
    train_indices = [i for i in range(101)]
    
    return images, poses, hwf_list, train_indices, val_indices, near_thresh, far_thresh