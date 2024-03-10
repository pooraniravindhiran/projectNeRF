import matplotlib.pyplot as plt 
import numpy
from PIL import Image
from dataset.load_blender import load_blender_data
from dataset.load_llff import load_llff_data
import torch
import numpy as np

LEGO_DATASET_DIR = '/Users/ajayravi/Desktop/project_nerf/data/lego/'
LLFF_DATASET_DIR = '/Users/ajayravi/Desktop/project_nerf/data/flower/'


def lego():
    half_res = True
    test_skip = 1
    white_bkgd = True
    images, poses, render_poses, hwf, i_split = load_blender_data(LEGO_DATASET_DIR,half_res, test_skip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, LEGO_DATASET_DIR)
    i_train, i_val, i_test = i_split
    # print(i_train.shape, i_val.shape, i_test.shape)
    # print(i_val)
    print(images.shape)
    
    near = 2.
    far = 6.

    if white_bkgd:
        images = images[...,:3]*images[...,-1:] + 1 * (1.-images[...,-1:])
    else:
        images = images[...,:3]


    images = torch.from_numpy(images)
    poses = torch.from_numpy(poses)
    # print(images.shape, poses.shape)
    plt.imshow(images[181])
    plt.show()

def llff():
    factor = 8
    llffhold = 8
    images, poses, bds, render_poses, i_test = load_llff_data(LLFF_DATASET_DIR, factor,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=False)
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, LLFF_DATASET_DIR)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if llffhold > 0:
        print('Auto LLFF holdout,', llffhold)
        i_test = np.arange(images.shape[0])[::llffhold]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])   

    print(i_train.shape, i_val.shape, i_test.shape)
    print(i_train, i_val, i_test) 
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    images = torch.from_numpy(images)
    poses = torch.from_numpy(poses)
    plt.imshow(images[0])
    plt.show()


# llff()
lego()