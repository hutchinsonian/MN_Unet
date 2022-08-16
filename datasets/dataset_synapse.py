import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def random_crop(image, label,output_size):
    h_off = np.random.randint(0, image.shape[0] - output_size[0])
    w_off = np.random.randint(0, image.shape[1] - output_size[1])
    image = image[h_off:h_off+output_size[0], w_off:w_off+output_size[1]]
    label = label[h_off:h_off+output_size[0], w_off:w_off+output_size[1]]
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        image, label = random_crop(image, label, self.output_size)
        # x, y = image.shape
        # if x != self.output_size[0] or y != self.output_size[1]:
        #     image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
        #     label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "test_train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npy')
            data = np.load(data_path, allow_pickle=True).item()
            image, label = data['image'], data['label']
            sample = {'image': image, 'label': label}
        else:
            vol_name = self.sample_list[idx].strip('\n').split('/')[-1]
            filepath = self.data_dir + '/' + vol_name
            img_itk = sitk.ReadImage(filepath)
            ct_array = sitk.GetArrayFromImage(img_itk)
            # 阈值截取
            ct_array[ct_array > 500] = 500
            ct_array[ct_array < -1000] = -1000
            ct_array = ct_array.astype(np.float32)
            ct_array = (ct_array-(-666.50824))/458.03006
            ct_array = (ct_array-(-0.7281002))/(2.5467942+0.7281002)
            spacing = np.array(img_itk.GetSpacing())
            origin = np.array(img_itk.GetOrigin())
            direction = np.array(img_itk.GetDirection())

            sample = {'image': ct_array, 'spacing': spacing, 'origin':origin,'direction':direction,'vol_name': vol_name}

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
