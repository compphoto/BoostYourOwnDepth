
from operator import getitem
from torchvision.transforms import Compose
from torchvision.transforms import transforms

# OUR
from utils import BoostDataset,write_depth
    

# PIX2PIX : MERGE NET
from pix2pix.options.test_options import TestOptions
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

import time
import os
import torch
import cv2
import numpy as np
import argparse
import warnings
warnings.simplefilter('ignore', np.RankWarning)

# select device
device = torch.device("cuda")
print("device: %s" % device)

# Global variables
pix2pixmodel = None
#factor = None
#whole_size_threshold = 3000  # R_max from the paper
#GPU_threshold = 1600 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted 



# MAIN PART OF OUR METHOD
def run(dataset, option):

    # Load merge network
    opt = TestOptions().parse()
    global pix2pixmodel
    pix2pixmodel = Pix2Pix4DepthModel(opt)
    pix2pixmodel.save_dir = option.checkpoints_dir +'/mergemodel'
    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()

    # Generating required directories
    result_dir = option.output_dir
    os.makedirs(result_dir, exist_ok=True)

   

    # Go through all images in input directory
    print("start processing")
    for image_ind, images in enumerate(dataset):
        print('processing image', image_ind, ':', images.name)
        print()

        # Load image from dataset
        low_res = images.low_res
        high_res = images.high_res
        input_resolution = low_res.shape
        print(input_resolution)

        

        # Generate the base estimate using the double estimation.
        whole_estimate = global_merge(low_res, high_res, option.pix2pixsize)
        
        path = os.path.join(result_dir, images.name)
        if option.output_resolution == 1:
            write_depth(path, cv2.resize(whole_estimate, (input_resolution[1], input_resolution[0]),
                                                   interpolation=cv2.INTER_CUBIC), bits=2,
                                        colored=option.colorize_results)
        else:
            write_depth(path, whole_estimate, bits=2, colored=option.colorize_results)

    print("finished")


# Generate a double-input depth estimation
def global_merge(low_res, high_res, pix2pixsize):
    # Generate the low resolution estimation
    estimate1 = low_res
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)
    depth_min = estimate1.min()
    depth_max = estimate1.max()
    

    if depth_max - depth_min > np.finfo("float").eps:
        estimate1 = (estimate1 - depth_min) / (depth_max - depth_min)
    else:
        estimate1 = 0


    # Generate the high resolution estimation
    estimate2 = high_res
    
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)
    depth_min = estimate2.min()
    depth_max = estimate2.max()

    #print(depth_min,depth_max)

    if depth_max - depth_min > np.finfo("float").eps:
        estimate2 = (estimate2 - depth_min) / (depth_max - depth_min)
    else:
        estimate2 = 0

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


if __name__ == "__main__":
    # Adding necessary input arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, required=True, help='input files directory '
                                                                    'Images can be .png .jpg .tiff')
    parser.add_argument('--output_dir', type=str, required=True, help='result dir. result depth will be png.'
                                                                      ' vides are JMPG as avi')
    parser.add_argument('--checkpoints_dir', type=str, required=True, help='weights file directory')                                                                  
    parser.add_argument('--output_resolution', type=int, default=1, required=False,
                        help='0 for results in maximum resolution 1 for resize to input size')
    parser.add_argument('--pix2pixsize', type=int, default=1024, required=False)  # Do not change it
    parser.add_argument('--colorize_results', action='store_true')
    parser.add_argument('--max_res', type=float, default=np.inf)

    # Check for required input
    option_, _ = parser.parse_known_args()
    print(option_)

    # Create dataset from input images
    dataset_ = BoostDataset(option_.data_dir, 'test')

    # Run pipeline
    run(dataset_, option_)