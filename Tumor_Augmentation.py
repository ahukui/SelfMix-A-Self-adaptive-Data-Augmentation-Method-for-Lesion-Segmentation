# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:17:22 2021

@author: 73239
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import random
import os
import SimpleITK as sitk
import sys, time
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm
import argparse
import os


def get_distance(f, spacing):
    """Return the signed distance."""

    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, -(dist_func(f, sampling=spacing)),
                        dist_func(1 - f, sampling=spacing))

    return distance


def get_head(img_path):
    temp = sitk.ReadImage(img_path)
    spacing = temp.GetSpacing()
    direction = temp.GetDirection()
    origin = temp.GetOrigin()

    return spacing, direction, origin


def copy_head_and_right_xyz(data, spacing, direction, origin):
    TrainData_new = data.astype('float32')
    TrainData_new = TrainData_new.transpose(2, 1, 0)
    TrainData_new = sitk.GetImageFromArray(TrainData_new)
    TrainData_new.SetSpacing(spacing)
    TrainData_new.SetOrigin(origin)
    TrainData_new.SetDirection(direction)

    return TrainData_new


def get_tumor_region(label_all):
    weights =np.zeros_like(label_all)
    for i in range(label_all.shape[0]):
        label = label_all[i]
        if np.sum(label)>0:
            mask = ndimage.label(label>0)[0]
            for _index in range(1, mask.max()+1):
                y, x = np.where(mask==_index)
                dismap = ndimage.morphology.distance_transform_edt(label[y.min()-1:y.max()+1, x.min()-1:x.max()+1])
                if np.max(dismap)<=1:
                    Nor_dis = dismap
                else:
                    Nor_dis = (dismap-np.min(dismap))/(np.max(dismap) -np.min(dismap))

                lam = np.random.uniform(0.7,1)

                weights[i, y.min()-1:y.max()+1, x.min()-1:x.max()+1] = ((1-lam)*Nor_dis + lam)*label[y.min()-1:y.max()+1, x.min()-1:x.max()+1]

    return weights
              

"""
==========================================
The input must be nii.gz which contains 
import header information such as spacing.
Spacing will affect the generation of the
signed distance.
=========================================
"""


def carvemix_generate_new_sample(image_a, image_b, label_a, label_b):
    spacing, direction, origin = get_head(image_a)

    target_a = nib.load(image_a).get_fdata()
    target_b = nib.load(image_b).get_fdata()
    label_a = nib.load(label_a).get_fdata()
    label_b = nib.load(label_b).get_fdata()
    label = np.copy(label_b)

    dis_array = get_distance(label, spacing)  # creat signed distance
    #     c = np.random.beta(1, 1)#[0,1]             #creat distance
    #     λl = np.min(dis_array)/2                   #λl = -1/2|min(dis_array)|
    #     λu = -np.min(dis_array)                    #λu = |min(dis_array)|
    #     lam = np.random.uniform(λl,λu,1)           #λ ~ U(λl,λu)
    c = np.random.beta(1, 1)  # [0,1] creat distance
    c = (c - 0.5) * 2  # [-1.1]
    if c > 0:
        lam = c * np.min(dis_array) / 2  # λl = -1/2|min(dis_array)|
    else:
        lam = c * np.min(dis_array)

    mask = (dis_array < lam).astype('float32')  # creat M

    new_target = target_a * (mask == 0) + target_b * mask
    new_label = label_a * (mask == 0) + label_b * mask

    new_target = copy_head_and_right_xyz(new_target, spacing, direction, origin)
    new_label = copy_head_and_right_xyz(new_label, spacing, direction, origin)
    mask = copy_head_and_right_xyz(mask, spacing, direction, origin)

    return new_target, new_label, mask, lam

def cutmix_generate_new_sample(image_a, image_b, label_a, label_b):
    spacing, direction, origin = get_head(image_a)

    target_a = nib.load(image_a).get_fdata()
    target_b = nib.load(image_b).get_fdata()
    label_a = nib.load(label_a).get_fdata()
    label_b = nib.load(label_b).get_fdata()
    label = np.copy(label_b)

    # dis_array = get_distance(label, spacing)  # creat signed distance
    # #     c = np.random.beta(1, 1)#[0,1]             #creat distance
    # #     λl = np.min(dis_array)/2                   #λl = -1/2|min(dis_array)|
    # #     λu = -np.min(dis_array)                    #λu = |min(dis_array)|
    # #     lam = np.random.uniform(λl,λu,1)           #λ ~ U(λl,λu)
    # c = np.random.beta(1, 1)  # [0,1] creat distance
    # c = (c - 0.5) * 2  # [-1.1]
    # if c > 0:
    #     lam = c * np.min(dis_array) / 2  # λl = -1/2|min(dis_array)|
    # else:
    #     lam = c * np.min(dis_array)

    #mask = np.copy(label_b)#(dis_array < lam).astype('float32')  # creat M

    new_target = target_a * (label_b == 0) + target_b * label_b
    new_label = np.clip(label_a  + label_b, 0, 1) 

    new_target = copy_head_and_right_xyz(new_target, spacing, direction, origin)
    new_label = copy_head_and_right_xyz(new_label, spacing, direction, origin)
    mask = copy_head_and_right_xyz(label_b, spacing, direction, origin)

    return new_target, new_label, mask#, lam

def mixup_generate_new_sample(image_a, image_b, label_a, label_b):
    spacing, direction, origin = get_head(image_a)

    target_a = nib.load(image_a).get_fdata()
    target_b = nib.load(image_b).get_fdata()
    label_a = nib.load(label_a).get_fdata()
    label_b = nib.load(label_b).get_fdata()
    label = np.copy(label_b)

    # dis_array = get_distance(label, spacing)  # creat signed distance
    # #     c = np.random.beta(1, 1)#[0,1]             #creat distance
    # #     λl = np.min(dis_array)/2                   #λl = -1/2|min(dis_array)|
    # #     λu = -np.min(dis_array)                    #λu = |min(dis_array)|
    # #     lam = np.random.uniform(λl,λu,1)           #λ ~ U(λl,λu)
    c = np.random.beta(1, 1)  # [0,1] creat distance
    # c = (c - 0.5) * 2  # [-1.1]
    # if c > 0:
    #     lam = c * np.min(dis_array) / 2  # λl = -1/2|min(dis_array)|
    # else:
    #     lam = c * np.min(dis_array)

    #mask = np.copy(label_b)#(dis_array < lam).astype('float32')  # creat M

    new_target = target_a * c + target_b * (1-c)
    new_label = np.clip(label_a*c  + label_b*(1-c), 0, 1) 

    new_target = copy_head_and_right_xyz(new_target, spacing, direction, origin)
    new_label = copy_head_and_right_xyz(new_label, spacing, direction, origin)
    mask = copy_head_and_right_xyz(label_b, spacing, direction, origin)

    return new_target, new_label, mask#, lam

def selfmix_generate_new_sample(image_a, image_b, label_a, label_b):
    spacing, direction, origin = get_head(image_a)

    target_a = nib.load(image_a).get_fdata()
    target_b = nib.load(image_b).get_fdata()
    label_a = nib.load(label_a).get_fdata()
    label_b = nib.load(label_b).get_fdata()

    label_b[label_a==1] = 0        
    weights = get_tumor_region(label_b)
    #print(np.min(weights), np.max(weights))
  
    bg = (1-label_b)*target_a

    fg = label_b*target_a * (1-weights) + label_b*target_b*(weights)
    new_target = bg + fg
    new_label = np.clip(label_a  + label_b, 0, 1) 

    new_target = copy_head_and_right_xyz(new_target, spacing, direction, origin)
    new_label = copy_head_and_right_xyz(new_label, spacing, direction, origin)
    #mask = copy_head_and_right_xyz(label_b, spacing, direction, origin)

    return new_target, new_label, weights, label_b





if __name__ == '__main__':
    
    
    image_a = './Task033_ATLAS2/imagesTr/train_R001_sub-r001s001_0000.nii.gz'
    image_b = './Task033_ATLAS2/imagesTr/train_R001_sub-r001s002_0000.nii.gz'
    label_a = './Task033_ATLAS2/labelsTr/train_R001_sub-r001s001.nii.gz'
    label_b = './Task033_ATLAS2/labelsTr/train_R001_sub-r001s002.nii.gz'
    

    new_target, new_label, mask, label_b = selfmix_generate_new_sample(image_a, image_b, label_a, label_b)
    sitk.WriteImage(new_target, 'IM.nii.gz')
    sitk.WriteImage(new_label, 'GT.nii.gz')

