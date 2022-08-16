import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk
import cv2

def random_rot_flip(t1_t1ce, flair_t2, WT, TC):
    # k = np.random.randint(0, 4)
    # t1_t1ce = np.rot90(t1_t1ce, k, axes=(1, 2))
    # flair_t2 = np.rot90(flair_t2, k, axes=(1, 2))
    # WT = np.rot90(WT, k)
    # TC = np.rot90(TC, k)
    axis = np.random.randint(0, 3)
    t1_t1ce = np.flip(t1_t1ce, axis=axis).copy()
    flair_t2 = np.flip(flair_t2, axis=axis).copy()
    # if axis!=0:
    WT = np.flip(WT, axis=axis-1).copy()
    TC = np.flip(TC, axis=axis-1).copy()
    return t1_t1ce, flair_t2, WT, TC


def random_rotate(t1_t1ce, flair_t2, WT, TC):
    angle = np.random.randint(-20, 20)
    t1_t1ce = ndimage.rotate(t1_t1ce, angle, order=0, reshape=False, axes=(1, 2))
    flair_t2 = ndimage.rotate(flair_t2, angle, order=0, reshape=False, axes=(1, 2))
    WT = ndimage.rotate(WT, angle, order=0, reshape=False)
    TC = ndimage.rotate(TC, angle, order=0, reshape=False)

    return t1_t1ce, flair_t2, WT, TC


def random_crop(t1_t1ce, flair_t2, WT, TC, output_size):
    # print('WT.shape[0] - output_size[0]:', WT.shape[0] - output_size[0])
    h_off = np.random.randint(0, WT.shape[0] - output_size[0])
    w_off = np.random.randint(0, WT.shape[1] - output_size[1])
    t1_t1ce = t1_t1ce[:, h_off:h_off + output_size[0], w_off:w_off + output_size[1]]
    flair_t2 = flair_t2[:, h_off:h_off + output_size[0], w_off:w_off + output_size[1]]
    WT = WT[h_off:h_off + output_size[0], w_off:w_off + output_size[1]]
    TC = TC[h_off:h_off + output_size[0], w_off:w_off + output_size[1]]
    # print('t1_t1ce after crop:', t1_t1ce.shape)
    # print('flair_t2 after crop:', flair_t2.shape)
    # print('WT after crop:', WT.shape)
    return t1_t1ce, flair_t2, WT, TC

def random_scale(t1_t1ce, flair_t2, WT, TC):
    scale = [1.25, 1.5]
    # print('t1_t1ce:', t1_t1ce.shape)
    # print('flair_t2:', flair_t2.shape)
    # print('WT:', WT.shape)
    # print('TC:', TC.shape)
    index = np.random.randint(0, 2)
    t1_t1ce = t1_t1ce.transpose(1,2,0)
    flair_t2 = flair_t2.transpose(1, 2, 0)
    t1_t1ce = cv2.resize(t1_t1ce, (0, 0), fx=scale[index], fy=scale[index],interpolation=cv2.INTER_LINEAR)
    flair_t2 = cv2.resize(flair_t2, (0, 0), fx=scale[index], fy=scale[index],interpolation=cv2.INTER_LINEAR)
    t1_t1ce = t1_t1ce.transpose(2,0,1)
    flair_t2 = flair_t2.transpose(2, 0, 1)
    WT = cv2.resize(WT, (0, 0), fx=scale[index], fy=scale[index],interpolation=cv2.INTER_NEAREST)
    TC = cv2.resize(TC, (0, 0), fx=scale[index], fy=scale[index],interpolation=cv2.INTER_NEAREST)
    # print('t1_t1ce after scale:', t1_t1ce.shape)
    # print('flair_t2 after scale:', flair_t2.shape)
    # print('WT after scale:', WT.shape)
    return t1_t1ce, flair_t2, WT, TC

def scale_224(t1_t1ce, flair_t2, WT, TC):
    index = np.random.randint(0, 2)
    t1_t1ce = t1_t1ce.transpose(1,2,0)
    flair_t2 = flair_t2.transpose(1, 2, 0)
    t1_t1ce = cv2.resize(t1_t1ce, (224, 224),interpolation=cv2.INTER_LINEAR)
    flair_t2 = cv2.resize(flair_t2, (224, 224),interpolation=cv2.INTER_LINEAR)
    t1_t1ce = t1_t1ce.transpose(2,0,1)
    flair_t2 = flair_t2.transpose(2, 0, 1)
    WT = cv2.resize(WT, (224, 224), interpolation=cv2.INTER_NEAREST)
    TC = cv2.resize(TC, (224, 224), interpolation=cv2.INTER_NEAREST)
    # print('t1_t1ce after scale:', t1_t1ce.shape)
    # print('flair_t2 after scale:', flair_t2.shape)
    # print('WT after scale:', WT.shape)
    return t1_t1ce, flair_t2, WT, TC


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        t1_t1ce, flair_t2, WT,TC = sample['t1_t1ce'],sample['flair_t2'],sample['WT'],sample['TC']
        t1_t1ce, flair_t2, WT, TC = scale_224(t1_t1ce, flair_t2, WT, TC)
        # print('t1_t1ce after scale:', t1_t1ce.shape)
        # print('flair_t2 after scale:', flair_t2.shape)
        # print('WT after scale:', WT.shape)

        if random.random() > 0.5:
            t1_t1ce, flair_t2, WT, TC = random_rot_flip(t1_t1ce, flair_t2, WT, TC)
        elif random.random() > 0.5:
            t1_t1ce, flair_t2, WT, TC = random_rotate(t1_t1ce, flair_t2, WT, TC)
        if random.random() > 0.5:
            t1_t1ce, flair_t2, WT, TC = random_scale(t1_t1ce, flair_t2, WT, TC)
            t1_t1ce, flair_t2, WT, TC = random_crop(t1_t1ce, flair_t2, WT, TC, self.output_size)

        # t1_t1ce_pre=[]
        # t1_t1ce_pre.append(t1_t1ce[0])
        # t1_t1ce_pre.append(t1_t1ce[3])
        # t1_t1ce_pre = np.asarray(t1_t1ce_pre)
        # t1_t1ce_now = []
        # t1_t1ce_now.append(t1_t1ce[1])
        # t1_t1ce_now.append(t1_t1ce[4])
        # t1_t1ce_now = np.asarray(t1_t1ce_now)
        # t1_t1ce_post = []
        # t1_t1ce_post.append(t1_t1ce[2])
        # t1_t1ce_post.append(t1_t1ce[5])
        # t1_t1ce_post = np.asarray(t1_t1ce_post)

        # flair_t2_pre = []
        # flair_t2_pre.append(flair_t2[0])
        # flair_t2_pre.append(flair_t2[3])
        # flair_t2_pre = np.asarray(flair_t2_pre)
        # flair_t2_now = []
        # flair_t2_now.append(flair_t2[1])
        # flair_t2_now.append(flair_t2[4])
        # flair_t2_now = np.asarray(flair_t2_now)
        # flair_t2_post = []
        # flair_t2_post.append(flair_t2[2])
        # flair_t2_post.append(flair_t2[5])
        # flair_t2_post = np.asarray(flair_t2_post)
        
        t1_t1ce = torch.from_numpy(t1_t1ce.astype(np.float32))
        flair_t2 = torch.from_numpy(flair_t2.astype(np.float32))
        # t1_t1ce_post = torch.from_numpy(t1_t1ce_post.astype(np.float32))
        # flair_t2_pre = torch.from_numpy(flair_t2_pre.astype(np.float32))
        # flair_t2_now = torch.from_numpy(flair_t2_now.astype(np.float32))
        # flair_t2_post = torch.from_numpy(flair_t2_post.astype(np.float32))

        WT = torch.from_numpy(WT.astype(np.float32))
        TC = torch.from_numpy(TC.astype(np.float32))

        # sample = {'t1_t1ce_pre': t1_t1ce_pre, 't1_t1ce_now': t1_t1ce_now,'t1_t1ce_post': t1_t1ce_post,
        #           'flair_t2_pre': flair_t2_pre, 'flair_t2_now': flair_t2_now,'flair_t2_post': flair_t2_post,
        #           'WT': WT.long(), 'TC': TC.long()}

        sample = {'t1_t1ce': t1_t1ce, 'flair_t2': flair_t2,
                  'WT': WT, 'TC': TC}

        return sample


class brats_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "train_19" or self.split == "train_20" or self.split == "train_20_224":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npy')
            data = np.load(data_path, allow_pickle=True).item()
            t1 = np.asarray(data['t1'])
            t1ce = np.asarray(data['t1ce'])
            t2 = np.asarray(data['t2'])
            flair = np.asarray(data['flair'])
            # label = data['label']
            # label[label == 4] = 3
            WT = np.asarray(data['WT'])
            TC = np.asarray(data['TC'])

            # t1_t1ce = []
            # t1_t1ce.append(t1)
            # t1_t1ce.append(t1ce)

            # flair_t2 = []
            # flair_t2.append(flair)
            # flair_t2.append(t2)
            t1_t1ce = np.concatenate([np.expand_dims(t1, 0), np.expand_dims(t1ce, 0)],axis=0)
            # print('t1: ', t1.shape)
            # print('t1ce: ', t1ce.shape)
            # print('t1_t1ce: ', t1_t1ce.shape)
            flair_t2 = np.concatenate([np.expand_dims(flair, 0),np.expand_dims(t2, 0)],axis=0)
            sample = {'t1_t1ce': t1_t1ce, 'flair_t2': flair_t2, 'WT': WT,'TC':TC}
        elif self.split == "train_19_and_20" :
            slice_name = self.sample_list[idx].strip('\n')
            data_path = slice_name + '.npy'
            data = np.load(data_path, allow_pickle=True).item()
            t1 = np.asarray(data['t1'])
            t1ce = np.asarray(data['t1ce'])
            t2 = np.asarray(data['t2'])
            flair = np.asarray(data['flair'])
            label = data['label']
            label[label == 4] = 3

            t1_t1ce = np.concatenate([t1, t1ce], axis=0)
            flair_t2 = np.concatenate([flair, t2], axis=0)
            sample = {'t1_t1ce': t1_t1ce, 'flair_t2': flair_t2, 'WT': label, 'TC': label}
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + '/' + vol_name
            data_list = sorted(os.listdir(filepath))

            img_itk = sitk.ReadImage(filepath + '/' + data_list[0])
            flair = sitk.GetArrayFromImage(img_itk)
            # print('flair1', flair.dtype)
            img_itk = sitk.ReadImage(filepath + '/' + data_list[1])
            t1 = sitk.GetArrayFromImage(img_itk)
            spacing = np.asarray(img_itk.GetSpacing())
            origin = np.asarray(img_itk.GetOrigin())
            direction = np.asarray(img_itk.GetDirection())

            # spacing = img_itk.GetSpacing()
            # origin = img_itk.GetOrigin()
            # direction = img_itk.GetDirection()
            # print('spacing1', spacing)
            img_itk = sitk.ReadImage(filepath + '/' + data_list[2])
            t1ce = sitk.GetArrayFromImage(img_itk)
            img_itk = sitk.ReadImage(filepath + '/' + data_list[3])
            t2 = sitk.GetArrayFromImage(img_itk)

            # flair = flair[2:146, 8:232, 8:232]
            # t1 = t1[2:146, 8:232, 8:232]
            # t1ce = t1ce[2:146, 8:232, 8:232]
            # t2 = t2[2:146, 8:232, 8:232]
            # print('flair1:', flair.shape)
            flair = flair.transpose(1, 2, 0)
            t1 = t1.transpose(1, 2, 0)
            t1ce = t1ce.transpose(1, 2, 0)
            t2 = t2.transpose(1, 2, 0)
            # print('flair2:', flair.shape)
            flair = cv2.resize(flair, (224, 224), interpolation=cv2.INTER_LINEAR)
            t1 = cv2.resize(t1, (224, 224), interpolation=cv2.INTER_LINEAR)
            t1ce = cv2.resize(t1ce, (224, 224), interpolation=cv2.INTER_LINEAR)
            t2 = cv2.resize(t2, (224, 224), interpolation=cv2.INTER_LINEAR)
            # print('flair3:', flair.shape)
            flair = flair.transpose(2, 0, 1)
            t1 = t1.transpose(2, 0, 1)
            t1ce = t1ce.transpose(2, 0, 1)
            t2 = t2.transpose(2, 0, 1)

            # print('flair2', flair.dtype)
            # print('flair4:', flair.shape)



            sample = {'flair': flair, 't1': t1, 't1ce': t1ce, 't2': t2,
                      'spacing': spacing, 'origin':origin,'direction':direction,'vol_name': vol_name}

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')

        return sample
