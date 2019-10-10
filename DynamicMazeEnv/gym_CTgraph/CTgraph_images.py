'''Create the image dataset to be used in the CT-graph'''
import numpy as np
from skimage import transform
import torch
import torchvision

class CTgraph_images:
    def __init__(self,conf_data):
        self.rnd = np.random.RandomState()
        self.rnd.seed(conf_data['image_dataset']['seed'])
        self.SMALL_ROTATION_ON_READ = conf_data['image_dataset']['small_rotation_on_read']
        self.NOISE_ON_IMAGES = conf_data['image_dataset']['noise_on_images_on_read']
        self.oneD = conf_data['image_dataset']['1D']
        self.high_value = 254 - self.NOISE_ON_IMAGES
        self.NR_OF_IMAGES = conf_data['image_dataset']['nr_of_images']
        if self.oneD:
            self.OBS_RES = [self.NR_OF_IMAGES+1,1]
        else:
            self.OBS_RES = [12,12] # resolution of input images
        self.image = np.zeros(((self.NR_OF_IMAGES,self.OBS_RES[0],self.OBS_RES[1])))
        for i in range(0,self.NR_OF_IMAGES):
            if self.oneD:
                self.image[i] = self.compute_highLevel_set(i)
            else:
                self.image[i] = self.compute_random_set(i)
        np.save('./',self.image)

    def getNoisyImage(self, img_nr):
        if self.oneD:
            noise = self.rnd.randint(0, self.NOISE_ON_IMAGES+1, self.OBS_RES[0]*self.OBS_RES[1]).reshape((self.OBS_RES[0], self.OBS_RES[1]))
            return self.image[img_nr] + noise
        else:
            noise = self.rnd.randint(0, self.NOISE_ON_IMAGES+1, self.OBS_RES[0]*self.OBS_RES[1]).reshape((self.OBS_RES[0], self.OBS_RES[1]))
            img = self.image[img_nr]
            return transform.rotate(img + noise, self.rnd.randint(0,self.SMALL_ROTATION_ON_READ)).astype(np.uint8)

    def add_reward_cue(self, img, reward):
        if self.oneD:
            # setting the last pixel to high
            img[self.OBS_RES[0]-1] = self.high_value * reward
            return img.astype(np.uint8)
        else:
            idx_st_h = int(self.OBS_RES[0]/2)
            idx_st_v = int(self.OBS_RES[1]/2)
            sz = 1
            img[idx_st_h-3*sz:idx_st_h+3*sz, idx_st_h-3*sz:idx_st_h+3*sz] = self.high_value * reward
            np.clip(img,0,254)
            return img.astype(np.uint8)

    def compute_random_set(self, img_nr):
        img = np.zeros((self.OBS_RES[0], self.OBS_RES[1]))
        # assuming parity
        idx_st_h = int(self.OBS_RES[0]/2)
        idx_st_v = int(self.OBS_RES[1]/2)
        sz = 1
        # creating the 4x4 checkers pattern
        small_img = self.rnd.randint(0, 3, size = 16).reshape(4,4)
        # upscaling
        img = np.kron(small_img, np.ones((3,3))) * self.high_value / 2

        img = transform.rotate(img,self.rnd.randint(0,359))
        return img.astype(np.uint8)

    def compute_highLevel_set(self, img_nr):
        img = np.zeros((self.OBS_RES[0], 1))
        img[img_nr][0] = self.high_value
        return img.astype(np.uint8)

    def nrOfImages(self):
        return self.NR_OF_IMAGES
