import torch
import numpy as np
from dataset.load_blender import load_blender_data
from dataset.load_llff import load_llff_data


def load_nerf_dataset(DATASET_TYPE, DATASET_DIR):
    """
    This function loads synthetic dataset - lego and real world - LLFF dataset
    """
    if DATASET_TYPE == 'lego':
        llffhold = 8
        images, poses, bds, render_poses, i_test = load_llff_data(DATASET_DIR, factor=8,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=False)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        near_thresh = 2.0 
        far_thresh = 6.0 
        if not isinstance(i_test, list):
            i_test = [i_test]

        if llffhold > 0:
            print('Auto LLFF holdout,', llffhold)
            i_test = np.arange(images.shape[0])[::llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])   
        print('Loaded llff', images.shape, render_poses.shape, hwf, DATASET_DIR)
        return near_thresh, far_thresh, hwf, images, poses, render_poses, i_train, i_val, i_test

    elif DATASET_TYPE == 'lego':
        half_res = True
        test_skip = 1
        white_bkgd = True
        images, poses, render_poses, hwf, i_split = load_blender_data(DATASET_DIR,half_res, test_skip)
        print('Loaded lego', images.shape, render_poses.shape, hwf, DATASET_DIR)
        i_train, i_val, i_test = i_split

        near_thresh = 2.0
        far_thresh = 6.0

        if white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
        
        return near_thresh, far_thresh, hwf, images, poses, render_poses, i_train, i_val, i_test
    else:
        print('Unknown Dataset')
