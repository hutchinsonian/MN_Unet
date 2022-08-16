import os.path as osp
import os
import random
import torch
import torch.optim as optim
from logger import Train_Logger
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm
from utils import DiceLoss, LossAverage, CosineScheduler
from torch.cuda.amp import autocast as autocast, GradScaler
from collections import OrderedDict
from torch.optim import lr_scheduler
from torchvision import transforms

def trainer_synapse(args, model, local_rank):
    from datasets.dataset_nifti_2d import RandomGenerator, nifti_dataset_2d
    base_lr = args.base_lr
    num_classes = args.num_classes

    batch_size = args.batch_size

    train_set = nifti_dataset_2d(args, mode="train",
                             transform=transforms.Compose([RandomGenerator(output_size=args.img_shape)]))
    if local_rank == 0:
        print('start train set')

    if local_rank == 0:
        print('start train sampler')
    train_sampler = distributed.DistributedSampler(train_set)

    if local_rank == 0:
        print("The length of train set is: {}".format(len(train_set)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id + local_rank)

    train_iter = DataLoader(train_set, batch_size=batch_size, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, sampler=train_sampler)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    iter_num = 0
    max_iterations = args.max_epochs * len(train_iter)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    save_path = osp.join('./results', args.output_dir)

    if local_rank == 0:
        if not osp.exists(save_path):
            os.mkdir(save_path)
        log = Train_Logger(save_path, "train_log")
        best = [0, 0]
    max_epoch = args.max_epochs
    scaler = GradScaler()
    trigger = 0
    for epoch_num in range(1, max_epoch + 1):
        train_sampler.set_epoch(epoch_num)
        #==================== train ====================#
        if local_rank == 0:
            print("=======Epoch:{}=======lr:{}".\
                  format(epoch_num, optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss = LossAverage()
        train_loss_ce = LossAverage()
        train_loss_dice = LossAverage()
        model.train()
        for i_batch, sample in tqdm(enumerate(train_iter), total=len(train_iter)):
            data = sample['ct']
            target = sample['label']
            data, target = data.float(), target.long()
            data, target = data.cuda(local_rank), target.cuda(local_rank)
            # 自动混合精度：加速前向训练
            with autocast():
                data = torch.unsqueeze(data, 1)
                outputs = model(data)

                loss_ce = ce_loss(outputs, target)
                loss_dice = dice_loss(outputs, target, softmax=True)
                loss = 0.3 * loss_ce + 0.7 * loss_dice

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_ = 1e-3 * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            lr = lr_
            iter_num = iter_num + 1

            train_loss.update(loss.item())
            train_loss_ce.update(loss_ce.item())
            train_loss_dice.update(loss_dice.item())

        if local_rank == 0:
            train_log = OrderedDict({'train_loss': train_loss.avg, 'train_dice': 1 - train_loss_dice.avg,
                                     'loss_ce': train_loss_ce.avg, 'loss_dice': train_loss_dice.avg})

        if local_rank == 0:
            log.update(epoch_num, train_log)
            state = {'net': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch_num}
            torch.save(state, osp.join(save_path, 'latest_net.pth'))
            trigger += 1
        # ==================== save best model ====================#
            if train_log['train_dice'] > best[1]:
                print('Saving best net')
                torch.save(state, osp.join(save_path, 'best_net.pth'))
                best[0] = epoch_num
                best[1] = train_log['train_dice']
                trigger = 0
            print('Best train dice at Epoch: {} | {}'.format(best[0], best[1]))
        # ==================== end save ====================#
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

    return "Training Finished!"
