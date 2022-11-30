import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import os
import random
from torchvision.transforms.functional import to_tensor
from utils import  save_to_json,load_from_json
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import torch
import math
import csv
from utils import save_to_json,load_from_json


rows = 288
cols = 288
PATCH_SIZE = 224
STRIDE_VAL = 256

def default_loader(path, channel=3):
    """
    :param path: image path
    :param channel: # image channel
    :return: image
    """
    if channel == 1:
        return Image.open(path).convert('L')
    else:
        assert (channel == 3)
        return Image.open(path).convert('RGB')  #

def OverlappingCropPatches(im, ref=None, patch_size=PATCH_SIZE,stride_val=STRIDE_VAL):
    """
    NonOverlapping Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :return: patches
    """
    rows, cols = im.size  #288
    y_loc = np.concatenate((np.arange(0, rows - patch_size, stride_val), np.array([rows - patch_size])), axis=0)
    num_y = len(y_loc)
    x_loc = np.concatenate((np.arange(0, cols - patch_size, stride_val), np.array([cols - patch_size])), axis=0)
    num_x = len(x_loc)
    num_patches = num_x * num_y
    #[ 0  8 16 24 32]

    patches = ()
    ref_patches = ()

    for i in y_loc:
        for j in x_loc:
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches = patches + (patch,)

            if ref is not None:
                ref_patch = to_tensor(ref.crop((j, i, j + patch_size, i + patch_size)))
                ref_patches = ref_patches + (ref_patch,)

    if ref is not None:
        return torch.stack(patches), torch.stack(ref_patches)
    else:
        return torch.stack(patches)


### REF, LQ1, LQ2, Label
class Dataset_Test_Compare(Dataset):
    """
       768 768
       """
    def __init__(self, test_root=None, test_csv=None, loader=default_loader, resize=224, stride_val=224, given_label=True):
        self.loader = loader
        self.test_root = test_root
        self.test_csv = test_csv
        self.resize = resize  # 256 768
        self.stride_val = stride_val  # 512 768
        self.test_contents = []
        self.given_label = given_label
        if self.test_csv is not None:
            with open(self.test_csv) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    file_list = [file for file in row[:3]]
                    if self.given_label:
                        file_list = file_list+[row[3]]
                    self.test_contents.append(file_list)
        
        self.contents = self.test_contents
    def __getitem__(self, index):

        o_ljz = self.loader(os.path.join(self.test_root,self.contents[index][0]))
        h, w = o_ljz.size
        # print(h, w)
        if h>self.resize and w>self.resize:
            o,a,b = self.loader( os.path.join(self.test_root,self.contents[index][0])),self.loader(os.path.join(self.test_root,self.contents[index][1])),self.loader(os.path.join(self.test_root,self.contents[index][2]))
            o = OverlappingCropPatches(o, patch_size=self.resize, stride_val=self.stride_val)
            a = OverlappingCropPatches(a, patch_size=self.resize, stride_val=self.stride_val)
            b = OverlappingCropPatches(b, patch_size=self.resize, stride_val=self.stride_val)
            label = self.contents[index][3] if self.given_label else 0
            return {'o': o, 'a': a, 'b': b, 'label': int(label)},{'o': self.contents[index][0], 'a': self.contents[index][1], 'b': self.contents[index][2]}
        else:
            o = self.loader( os.path.join(self.test_root,self.contents[index][0]))
            a = self.loader(os.path.join(self.test_root,self.contents[index][1]))
            b = self.loader(os.path.join(self.test_root,self.contents[index][2]))
            crop_fuc = transforms.CenterCrop((self.resize if h >self.resize else h, self.resize if w >self.resize else w))
            o,a,b = crop_fuc(o),crop_fuc(a),crop_fuc(b)
            o, a, b = to_tensor(o), to_tensor(a), to_tensor(b)
            o,a,b = o.unsqueeze(0),a.unsqueeze(0),b.unsqueeze(0)
            label = self.contents[index][3] if self.given_label else 0
            return {'o': o, 'a': a, 'b': b, 'label': int(label)},{'o': self.contents[index][0], 'a': self.contents[index][1], 'b': self.contents[index][2]}

    def __len__(self):
        return len(self.contents)


### REF, LQ1
class Dataset_Test(Dataset):
    """
       768 768
       """
    def __init__(self, test_root=None, test_csv=None, loader=default_loader, resize=224, stride_val=224, given_label=False):
        self.loader = loader
        self.test_root = test_root
        self.test_csv = test_csv
        self.resize = resize  # 256 768
        self.stride_val = stride_val  # 512 768
        self.test_contents = []
        self.given_label = given_label
        with open(self.test_csv) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                file_list = [file for file in row[:2]]
                if self.given_label:
                    file_list = file_list+[row[2]]
                self.test_contents.append(file_list)
        print(self.test_contents)
        self.contents = self.test_contents
    def __getitem__(self, index):

        o_ljz = self.loader(os.path.join(self.test_root,self.contents[index][0]))
        h, w = o_ljz.size
        # print(h, w)
        if h>self.resize and w>self.resize:
            o,a = self.loader( os.path.join(self.test_root,self.contents[index][0])),self.loader(os.path.join(self.test_root,self.contents[index][1]))
            o = OverlappingCropPatches(o, patch_size=self.resize, stride_val=self.stride_val)
            a = OverlappingCropPatches(a, patch_size=self.resize, stride_val=self.stride_val)
           
            label = self.contents[index][2] if self.given_label else 0
            return {'o': o, 'a': a, 'label': int(label)},{'o': self.contents[index][0], 'a': self.contents[index][1]}
        else:
            o = self.loader( os.path.join(self.test_root,self.contents[index][0]))
            a = self.loader(os.path.join(self.test_root,self.contents[index][1]))
            crop_fuc = transforms.CenterCrop((self.resize if h >self.resize else h, self.resize if w >self.resize else w))
            o,a = crop_fuc(o),crop_fuc(a)
            o, a = to_tensor(o), to_tensor(a)
            o,a = o.unsqueeze(0),a.unsqueeze(0)
            label = self.contents[index][2] if self.given_label else 0
            return {'o': o, 'a': a, 'label': int(label)},{'o': self.contents[index][0], 'a': self.contents[index][1]}

    def __len__(self):
        return len(self.contents)
