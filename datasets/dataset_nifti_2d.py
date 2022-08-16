import random
import torch
from scipy import ndimage
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
import cv2


def random_rot_flip(ct, label):
    axis = np.random.randint(0, 2)
    ct = np.flip(ct, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return ct, label


def random_rotate(ct, label):
    angle = np.random.randint(-20, 20)
    ct = ndimage.rotate(ct, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return ct, label


def random_crop(ct, label, output_size):
    h_off = np.random.randint(0, label.shape[0] - output_size[0])
    w_off = np.random.randint(0, label.shape[1] - output_size[1])
    ct = ct[h_off:h_off + output_size[0], w_off:w_off + output_size[1]]
    label = label[h_off:h_off + output_size[0], w_off:w_off + output_size[1]]
    return ct, label


def random_scale(ct, label):
    scale = [1.25, 1.5]
    index = np.random.randint(0, 2)
    ct = cv2.resize(ct, (0, 0), fx=scale[index], fy=scale[index], interpolation=cv2.INTER_NEAREST)
    label = cv2.resize(label, (0, 0), fx=scale[index], fy=scale[index], interpolation=cv2.INTER_NEAREST)
    return ct, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.mean = 0.
        self.variance = 15.

    def __call__(self, sample):
        ct, label = sample['ct'], sample['label']

        if random.random() > 0.5:
            ct, label = random_rot_flip(ct, label)
        elif random.random() > 0.5:
            ct, label = random_rotate(ct, label)
        if random.random() > 0.5:
            ct, label = random_scale(ct, label)
            ct, label = random_crop(ct, label, self.output_size)

        ct = torch.from_numpy(ct.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'ct': ct, 'label': label}
        return sample


class nifti_dataset_2d(Dataset):
    def __init__(self, args, mode, transform=None):
        self.transform = transform  # using transform in torch!
        self.mode = mode
        self.args = args

        if mode == "train":
            filename = "train_path_list.txt"
        elif mode == "valid":
            filename = "val_path_list.txt"

        self.filename_list = open(osp.join(args.processed_dataset_path, filename)).readlines()

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):

        slice_name = self.filename_list[idx].strip('\n')
        data_path = osp.join(slice_name + '.npy')
        data = np.load(data_path, allow_pickle=True).item()
        ct = np.asarray(data['ct'])
        label = np.asarray(data['label'])

        sample = {'ct': ct, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        # sample['case_name'] = self.filename_list[idx].strip('\n')

        return sample