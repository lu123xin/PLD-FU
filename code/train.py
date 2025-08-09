import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils import ramps, losses, metrics, test_patch
from utils.losses import dice_loss
from dataloaders.dataset import *
from networks.net_factory import net_factory


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def augment(volume):
    if random.random() > 0.5:
        volume = torch.flip(volume, dims=[3])  
    if random.random() > 0.5:
        volume = torch.rot90(volume, k=1, dims=[3, 4])  # H-D 
    return volume


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/root/PLD_FU/',
                    help='Name of Dataset')
parser.add_argument('--exp', type=str, default='MCNet', help='exp_name')
parser.add_argument('--model', type=str, default='mcnet3d_v1', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')

args = parser.parse_args()

snapshot_path = "/root/PLD_FU/data/{}_{}_{}_labeled/{}".format(
    args.dataset_name, args.exp,
    args.labelnum, args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path + 'data/Pancreas/'
    args.max_samples = 62
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    # 创建保存路径及代码备份
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(sys.argv[0])

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model = model.cuda()

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train,
                             batch_sampler=batch_sampler,
                             num_workers=4,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()

         
            volume_batch_aug = augment(volume_batch)

            
            model.eval()
            with torch.no_grad():
                preds_orig = model(volume_batch)  # [B, C, W, H, D]
                preds_aug = model(volume_batch_aug)
            model.train()

            preds_orig_prob = F.softmax(preds_orig, dim=1)
            preds_aug_prob = F.softmax(preds_aug, dim=1)

           
            _, pseudo_outputs_1 = torch.max(preds_orig_prob, dim=1)
            _, pseudo_outputs_2 = torch.max(preds_aug_prob, dim=1)
            mtx_bool_conflict = (pseudo_outputs_1 != pseudo_outputs_2)  # bool mask [B, W, H, D]
            mask_phi_u_a = mtx_bool_conflict.float()

          
            entropy_orig = -torch.sum(preds_orig_prob * torch.log(preds_orig_prob + 1e-10), dim=1)  # [B, W, H, D]
            weights = torch.exp(-entropy_orig) * mask_phi_u_a

    
            diff = preds_aug_prob - preds_orig_prob  # [B, C, W, H, D]
            diff_sq = diff.pow(2).sum(dim=1)          # [B, W, H, D]
            loss_consistency = (weights * diff_sq).sum() / (weights.sum() + 1e-6)

            loss_seg_dice = dice_loss(preds_orig_prob[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = args.lamda * loss_seg_dice + consistency_weight * loss_consistency

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            logging.info('Iteration %d: Total loss=%.4f, Dice loss=%.4f, Consistency loss=%.4f' %
                         (iter_num, loss.item(), loss_seg_dice.item(), loss_consistency.item()))

            writer.add_scalar('Loss/total', loss.item(), iter_num)
            writer.add_scalar('Loss/dice', loss_seg_dice.item(), iter_num)
            writer.add_scalar('Loss/consistency', loss_consistency.item(), iter_num)

            # --- 验证与保存模型 ---
            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=18,
                                                          stride_z=4,
                                                          dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=16,
                                                          stride_z=16,
                                                          dataset_name='Pancreas_CT')

                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("Saved best model to {}".format(save_mode_path))

                writer.add_scalar('Validation/Dice', dice_sample, iter_num)
                writer.add_scalar('Validation/Best_Dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_{}.pth'.format(iter_num))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("Saved final model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
