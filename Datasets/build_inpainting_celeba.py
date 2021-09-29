import os
import matplotlib.pyplot as plt
import sys
datasets = os.path.dirname(os.path.abspath(__file__)).strip()
sys.path.append(datasets)

inp = r".\img_align_celeba"
out128 = r".\celeba_inpainted"
out256 = r".\celeba_256"
from tqdm import tqdm
import cv2
from numba import prange
import numpy as np
mkdirs = lambda x: os.path.exists(x) or os.makedirs(x)
TEST_SIZE = 30000
files = os.listdir(inp)
length_dataset = len(files)
def inpainting(image, cords):
    (x1,y1), (x2,y2) = cords
    image[y1:y2, x1:x2] = 0
    return image

def build_inpainting_dataset():
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

            y = np.random.randint(45, 60)
            x = np.random.randint(10, 65)
            w = np.random.randint(60, 80)
            h = np.random.randint(12, 30)
            im_new_inpainted = inpainting(im_new_128, [(x,y),(x+w,y+h)])
            aligned_im = np.zeros( (im_new_inpainted.shape[0],im_new_inpainted.shape[1] * 2,3))
            aligned_im[0:im_new_inpainted.shape[0], 0:im_new_inpainted.shape[1]] = im_new_inpainted
            aligned_im[0:im_new_inpainted.shape[0], im_new_inpainted.shape[1]:] = im_new_128
            if i <(length_dataset-TEST_SIZE): # set train set
                cv2.imwrite(os.path.join(out128, 'trainA', f), im_new_inpainted)
                cv2.imwrite(os.path.join(out128, 'trainB', f), im_new_128) # output
                cv2.imwrite(os.path.join(out128, 'train', f), aligned_im) # output
            else:
                cv2.imwrite(os.path.join(out128, 'valA', f), im_new_inpainted)
                cv2.imwrite(os.path.join(out128, 'valB', f), im_new_128)  # output
                cv2.imwrite(os.path.join(out128, 'val', f), aligned_im)  # output

if __name__ =='__main__':
    build_inpainting_dataset()
