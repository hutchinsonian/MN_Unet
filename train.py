import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import trainer_synapse
import config
import torch.nn as nn
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
local_rank = int(os.environ["LOCAL_RANK"])

def init_net(net):
    if isinstance(net, nn.Conv3d) or isinstance(net, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(net.weight.data, 0.2)
        nn.init.constant_(net.bias.data, 0)

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        
if __name__ == "__main__":
    args = config.args
    print('args')
    torch.distributed.init_process_group('nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    print('distributed')

    from networks import mn_unet

    print('before init_seeds')
    init_seeds(args.seed + local_rank)
    print('before net')
    net = mn_unet.MN_Unet(num_classes=args.num_classes).cuda(local_rank)

    print('net')
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net).cuda(local_rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    # net.apply(init_net)

    print('train')
    trainer = trainer_synapse
    trainer(args, net, local_rank)
