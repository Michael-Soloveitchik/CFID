import os
import matplotlib.pyplot as plt
import sys
CFID_dir = os.path.dirname(os.path.abspath(__file__)).strip()

inp = r".\Datasets\img_align_celeba"
out128 = r".\Datasets\celeba_rotation"
out256 = r".\Datasets\celeba_256"
from tqdm import tqdm
import cv2
from numba import prange
import numpy as np
mkdirs = lambda x: os.path.exists(x) or os.makedirs(x)
TEST_SIZE = 30000
files = os.listdir(inp)
length_dataset = len(files)
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def build_rotation_90_dataset():
    for out in [out128]:
        mkdirs(out)
        mkdirs(os.path.join(out, 'trainA'))
        mkdirs(os.path.join(out, 'trainB'))
        mkdirs(os.path.join(out, 'train'))
        mkdirs(os.path.join(out, 'valA'))
        mkdirs(os.path.join(out, 'valB'))
        mkdirs(os.path.join(out, 'val'))

        for i in tqdm(prange(len(files))):
            f=files[i]
            im = cv2.imread(os.path.join(inp,f))
            n,m,_ = im.shape
            l = (n-m)//2
            u = (m-m)//2
            im_new_128 = cv2.resize(im[l:l+m,u:u+m], (128,128)) #ground truth

            ang = np.random.randint(80,100)
            im_new_128_rot_ang = rotate_image(im_new_128,ang)

            aligned_im = np.zeros( (im_new_128_rot_ang.shape[0],im_new_128_rot_ang.shape[1] * 2,3))
            aligned_im[0:im_new_128_rot_ang.shape[0], 0:im_new_128_rot_ang.shape[1]] = im_new_128_rot_ang
            aligned_im[0:im_new_128_rot_ang.shape[0], im_new_128_rot_ang.shape[1]:] = im_new_128
            if i <(length_dataset-TEST_SIZE): # set train set
                cv2.imwrite(os.path.join(out128, 'trainA', f), im_new_128_rot_ang)
                cv2.imwrite(os.path.join(out128, 'trainB', f), im_new_128) # output
                cv2.imwrite(os.path.join(out128, 'train', f), aligned_im) # output
            else:
                cv2.imwrite(os.path.join(out128, 'valA', f), im_new_128_rot_ang)
                cv2.imwrite(os.path.join(out128, 'valB', f), im_new_128)  # output
                cv2.imwrite(os.path.join(out128, 'val', f), aligned_im)  # output

if __name__ =='__main__':
    build_rotation_90_dataset()
