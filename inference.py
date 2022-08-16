import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import SimpleITK as sitk
import os.path as osp
import config
import os
import re

local_rank = int(os.environ["LOCAL_RANK"])

# def get_filename(raw_name):
#     save_dir = raw_name.split('-')[-1].split('_')[0:-1]
#     if len(save_dir) == 2:
#         save_dir = str(save_dir[0][1:]) + '_' + save_dir[1] + '.nii.gz'
#     elif len(save_dir) > 2:
#         assert ("error submission")
#     else:
#         save_dir = save_dir[0][1:] + '.nii.gz'
#     return save_dir

def get_filename(raw_name):
    return raw_name.split('-')[-1]

if __name__ == "__main__":
    args = config.args
    file_list = os.listdir("Validation_lits")
    torch.distributed.init_process_group('nccl', init_method='env://')
    torch.cuda.set_device(local_rank)

    from networks import nat, Unet
    net = nat.NAT(num_classes=args.num_classes).cuda(local_rank)
    # net = Unet.Unet(num_classes=args.num_classes).cu`da()

    ckpt = torch.load('./results/lesion/best_net.pth', map_location=torch.device("cpu"))
    net.load_state_dict(ckpt['net'])
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda(local_rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    net.eval()

    with torch.no_grad():
        for data in tqdm(file_list, total=len(file_list)):
            # print('data:', data)
            ct = sitk.ReadImage(osp.join('Validation_lits', data), sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)

            spacing = ct.GetSpacing()
            origin = ct.GetOrigin()
            direction = ct.GetDirection()

            # ct_array = (ct_array - np.mean(ct_array)) / np.std(ct_array)

            ct_array[ct_array > 200] = 200
            ct_array[ct_array < -200] = -200

            ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
            ct_array = ct_array.cuda()
            ct_array = torch.unsqueeze(ct_array, 0)

            for data_slice in range(ct_array.size()[2]):

                pred_img = net(ct_array[:, :, data_slice, :, :]).argmax(dim=1)
                # pred_img, _ = net(ct_array[:, :, data_slice, :, :])
                # pred_img = pred_img.argmax(dim=1)
                # print('pred', pred_img.size())
                if data_slice == 0:
                    pred_img_whole = torch.unsqueeze(pred_img, 0)
                else:
                    pred_img_whole = torch.cat((pred_img_whole, torch.unsqueeze(pred_img, 0)), dim=1)
                # print('pred_img_whole:', pred_img_whole.shape)

            pred_img_whole = torch.squeeze(pred_img_whole, 0)
            pred_img_whole = np.asarray(pred_img_whole.cpu().numpy(), dtype='uint8')

            pred_img_whole = sitk.GetImageFromArray(pred_img_whole)
            pred_img_whole.SetSpacing(spacing)
            pred_img_whole.SetOrigin(origin)
            pred_img_whole.SetDirection(direction)
            if not osp.exists('inference_result'): os.mkdir('inference_result')
            # data_path = re.findall("\d+\.?\d*", data)[-1][-3:]
            if local_rank == 0:
                sitk.WriteImage(pred_img_whole, osp.join('inference_result', 'test-segmentation-' + get_filename(data)))



