import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import skimage.measure
import glob

# miscellaneous function for reading, writing and processing rgb and depth images.


def resizewithpool(img, size):
    i_size = img.shape[0]
    n = int(np.floor(i_size/size))

    out = skimage.measure.block_reduce(img, (n, n), np.max)
    return out


def showimage(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255.0
    #if img.ndim == 2:
    #    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img.astype(np.float32)


def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.15*size[0]):size[0] - int(0.15*size[0]), int(0.15*size[1]): size[1] - int(0.15*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask


def impatch(image, rect):
    # Extract the given patch pixels from a given image.
    w1 = rect[0]
    h1 = rect[1]
    w2 = w1 + rect[2]
    h2 = h1 + rect[3]
    image_patch = image[h1:h2, w1:w2]
    return image_patch


def getGF_fromintegral(integralimage, rect):
    # Computes the gradient density of a given patch from the gradient integral image.
    x1 = rect[1]
    x2 = rect[1]+rect[3]
    y1 = rect[0]
    y2 = rect[0]+rect[2]
    value = integralimage[x2, y2]-integralimage[x1, y2]-integralimage[x2, y1]+integralimage[x1, y1]
    return value


def rgb2gray(rgb):
    # Converts rgb to gray
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def calculateprocessingres(img, basesize, confidence=0.1, scale_threshold=3, whole_size_threshold=3000):
    # Returns the R_x resolution described in section 5 of the main paper.

    # Parameters:
    #    img :input rgb image
    #    basesize : size the dilation kernel which is equal to receptive field of the network.
    #    confidence: value of x in R_x; allowed percentage of pixels that are not getting any contextual cue.
    #    scale_threshold: maximum allowed upscaling on the input image ; it has been set to 3.
    #    whole_size_threshold: maximum allowed resolution. (R_max from section 6 of the main paper)

    # Returns:
    #    outputsize_scale*speed_scale :The computed R_x resolution
    #    patch_scale: K parameter from section 6 of the paper

    # speed scale parameter is to process every image in a smaller size to accelerate the R_x resolution search
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))

    gray = rgb2gray(img)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

    # thresholding the gradient map to generate the edge-map as a proxy of the contextual cues
    m = grad.min()
    M = grad.max()
    middle = m + (0.4 * (M - m))
    grad[grad < middle] = 0
    grad[grad >= middle] = 1

    # dilation kernel with size of the receptive field
    kernel = np.ones((int(basesize/speed_scale), int(basesize/speed_scale)), np.float)
    # dilation kernel with size of the a quarter of receptive field used to compute k
    # as described in section 6 of main paper
    kernel2 = np.ones((int(basesize / (4*speed_scale)), int(basesize / (4*speed_scale))), np.float)

    # Output resolution limit set by the whole_size_threshold and scale_threshold.
    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))

    outputsize_scale = basesize / speed_scale
    for p_size in range(int(basesize/speed_scale), int(threshold/speed_scale), int(basesize / (2*speed_scale))):
        grad_resized = resizewithpool(grad, p_size)
        grad_resized = cv2.resize(grad_resized, (p_size, p_size), cv2.INTER_NEAREST)
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0

        dilated = cv2.dilate(grad_resized, kernel, iterations=1)
        meanvalue = (1-dilated).mean()
        if meanvalue > confidence:
            break
        else:
            outputsize_scale = p_size

    grad_region = cv2.dilate(grad_resized, kernel2, iterations=1)
    patch_scale = grad_region.mean()

    return int(outputsize_scale*speed_scale), patch_scale



def write_depth(path, depth, bits=1 , colored=False):
    """Write depth map to png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    
    if colored == True:
        bits = 1

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1
    
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1 or colored:
        out = out.astype("uint8")
        if colored:
            out = cv2.applyColorMap(out,cv2.COLORMAP_INFERNO)
        cv2.imwrite(path+'.png', out)
    elif bits == 2:
        cv2.imwrite(path+'.png', out.astype("uint16"))

    return


class Images:
    def __init__(self, root_dir, files, index):
        self.root_dir = root_dir
        name = files[index]
        self.rgb_image = read_image(os.path.join(self.root_dir, name))
        name = name.replace(".jpg", "")
        name = name.replace(".png", "")
        name = name.replace(".jpeg", "")
        self.name = name

class Depths:
    def __init__(self, lr_dir, hr_dir, files, index):
        #self.root_dir = root_dir
        name = files[index]
        self.low_res = read_image(os.path.join(lr_dir, name))
        self.high_res = read_image(os.path.join(hr_dir, name))
        name = name.replace(".jpg", "")
        name = name.replace(".png", "")
        name = name.replace(".jpeg", "")
        self.name = name


class BoostDataset:
    def __init__(self, root_dir, subsetname):
        self.dataset_dir = root_dir
        self.subsetname = subsetname
        self.lr_depth_dir = os.path.join(root_dir,'low-res')
        self.hr_depth_dir = os.path.join(root_dir,'high-res')
        self.files = sorted(glob.glob(os.path.join(self.lr_depth_dir, '*')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return Depths(self.lr_depth_dir,self.hr_depth_dir, self.files, index)
