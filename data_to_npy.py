import os.path as osp
import os
import SimpleITK as sitk
from scipy import ndimage
import random
import numpy as np
import config

class Data_process:
    def __init__(self, raw_dataset_path, processed_dataset_path, args):
        # 数据集文件路径
        self.raw_root_path = raw_dataset_path
        # 处理后数据集存放路径
        self.processed_path = processed_dataset_path
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice
        self.index = 0

    def process_data(self):
        # 创建保存数据的文件夹
        if not os.path.exists(self.processed_path):
            os.makedirs(osp.join(self.processed_path, "case"))

        # ct
        file_list = os.listdir(osp.join(self.raw_root_path, 'ct'))
        num_file = len(file_list)
        print('totle numbers of samples is : ', num_file)
        for ct_file, i in zip(file_list, range(num_file)):
            print("==== {} | {}/{} ====".format(ct_file, i + 1, num_file))
            ct_path = osp.join(self.raw_root_path, 'ct', ct_file)
            seg_path = osp.join(self.raw_root_path, 'label', ct_file.replace('_ct', '_seg'))
            self.process(ct_path, seg_path)



    def process(self, ct_path, seg_path):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:", ct_array.shape, seg_array.shape)

        # ct_array[ct_array > self.upper] = self.upper
        # ct_array[ct_array < self.lower] = self.lower

        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        start_slice = max(0, start_slice - self.expand_slice)
        end_slice = min(ct_array.shape[0] - 1, end_slice + self.expand_slice)

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        print("Preprocessed shape:", ct_array.shape, seg_array.shape)

        ct_array = (ct_array - np.mean(ct_array)) / np.std(ct_array)
        # seg_array = (seg_array - np.mean(seg_array)) / np.std(seg_array)

        save_case_path = osp.join(self.processed_path, "case")

        for i in range(ct_array.shape[0]):
            save_case = {}
            save_case['ct'] = ct_array[i]
            save_case['label'] = seg_array[i]
            np.save(osp.join(save_case_path, "case"+str(self.index)), save_case)

            f = open(osp.join(self.processed_path, "train_path_list.txt"), 'a')
            fpath = osp.join(save_case_path, "case" + str(self.index))
            f.write(fpath + '\n')
            f.close()
            self.index += 1

        print("Current number of sample is :", self.index)

if __name__ == '__main__':
    raw_dataset_path = './datasets/train'
    processed_dataset_path = './processed_datasets'

    args = config.args
    tool = Data_process(raw_dataset_path, processed_dataset_path, args)
    tool.process_data()
