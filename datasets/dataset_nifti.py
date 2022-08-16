import random
import torch
import SimpleITK as sitk
import os.path as osp
import numpy as np
from torch.utils.data import Dataset


class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)


class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop:
    def __init__(self, slices):
        self.slices = slices

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __call__(self, img, mask):

        ss, es = self._get_range(mask.size(1), self.slices)

        # print(self.shape, img.shape, mask.shape)
        tmp_img = torch.zeros((img.size(0), self.slices, img.size(2), img.size(3)))
        tmp_mask = torch.zeros((mask.size(0), self.slices, mask.size(2), mask.size(3)))
        tmp_img[:, :es - ss] = img[:, ss:es]
        tmp_mask[:, :es - ss] = mask[:, ss:es]
        return tmp_img, tmp_mask


class Center_Crop:
    def __init__(self, base, max_size):
        self.base = base  # base默认取16，因为4次下采样后为1
        self.max_size = max_size
        if self.max_size % self.base:
            self.max_size = self.max_size - self.max_size % self.base

    def __call__(self, img, label):
        if img.size(1) < self.base:
            return None
        slice_num = img.size(1) - img.size(1) % self.base
        slice_num = min(self.max_size, slice_num)

        left = img.size(1) // 2 - slice_num // 2
        right = img.size(1) // 2 + slice_num // 2

        crop_img = img[:, left:right]
        crop_label = label[:, left:right]
        return crop_img, crop_label


class RandomCrop3D():
    def __init__(self, img_sz, crop_sz):
        b, c, h, w, d = img_sz
        assert (h, w, d) > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)

    def __call__(self, x):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return (None, None)

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]


class niftiDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

        if mode == "train":
            filename = "train_path_list.txt"
            self.transforms = Compose([
                RandomCrop(self.args.crop_size),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
            ])
        elif mode == "valid":
            self.transforms = Compose([Center_Crop(base=16, max_size=args.val_crop_max_size)])
            filename = "val_path_list.txt"

        self.filename_list = self.load_file_name_list(osp.join(args.processed_dataset_path, filename))

    def __getitem__(self, index):
        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)
        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list