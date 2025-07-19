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
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils import ramps, losses, metrics, test_patch
from utils.losses import  dice_loss
from dataloaders.dataset import *
from networks.net_factory import net_factory


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/mutual-learning-with-reliable-pseudo-labels/',
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

snapshot_path = "/root/autodl-tmp/mutual-learning-with-reliable-pseudo-labels/data/{}_{}_{}_labeled/{}".format(
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
var_param_conflict_weight=0.5
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    ## make logger file
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
    logging.info("{} itertations per epoch".format(len(trainloader)))
    pixel_criterion = losses.ce_loss_mask
    consistency_criterion = nn.CrossEntropyLoss(reduction='none')
    dice_loss = losses.Binary_dice_loss

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # print("train__-140_",volume_batch.shape)
            # print("train_____141__",label_batch.shape)

            model.train()
            for num in range(3): #利用解码器差异性更新编码器，mask存储的是各个层的特征图
                if num == 0:
                    outputs, outfeats,mask,stage_out1= model(volume_batch, [])
                else:
                    outputs, outfeats ,mask,stage_out1= model(volume_batch, en)

                consistency_weight = get_current_consistency_weight(iter_num // 150)

                en = []
                for idx in range(len( mask[0])):
                    mask1 =  mask[0][idx].detach()
                    mask2 =  mask[1][idx].detach()
                    # mask1 = sharpening(mask1)
                    # mask2 = sharpening(mask2)
                    en.append(1e-3 * (mask1 - mask2))
                #print('en',en[3].shape)
            # outputs, outfeats = model(volume_batch)
            # outputs, outfeats, stage_out1= model(volume_batch)
                num_outputs = len(outputs)

                y_all = torch.zeros((num_outputs,) + outputs[0].shape).cuda()#y_all形状：（2，outputs[0].shape）#初始化一个张量来存储所有输出的概率。

                # loss_seg = 0
                loss_seg_dice = 0
                #for idx in range(num_outputs):
                y1 = outputs[0]
                y2= outputs[1]
                y_prob1 = F.softmax(y1, dim=1)
                y_prob2 = F.softmax(y2, dim=1)
                out5, out4, out3, out2 = stage_out1[0], stage_out1[1], stage_out1[2], stage_out1[3]
                #out1_soft = F.softmax(out1, dim=1)
                out2_soft = F.softmax(out2, dim=1)
                out3_soft = F.softmax(out3, dim=1)
                out4_soft = F.softmax(out4, dim=1)
                out5_soft = F.softmax(out5, dim=1)

                #outputs_soft1 = F.softmax(outputs1, dim=1)
                #outputs_soft2 = F.softmax(outputs2, dim=1)

                # calculate the loss
                # supervised loss
                #print('y_prob1[:labeled_bs, 1, ...]',y_prob1[:labeled_bs, 1, ...].shape)
                loss_sup1 = losses.dice_loss(y_prob1[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1) #监督损失
                loss_sup2 = losses.dice_loss(y_prob2[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)
                # loss_sup2 = torch.mean(ce_loss(outputs2[:labeled_bs], label_batch[:labeled_bs]))
                # loss_sup2 = -torch.sum(label_batch[:labeled_bs]*(outputs_soft2[:labeled_bs, 1, :, :, :]+1e-6).log()) / (torch.sum(label_batch[:labeled_bs])+ 1e-6) \
                # -torch.sum((1.-label_batch[:labeled_bs])*(1.-outputs_soft2[:labeled_bs, 1, :, :, :]+1e-6).log()) / (torch.sum(1.-label_batch[:labeled_bs]) + 1e-6)
                loss_sup = loss_sup1 + loss_sup2
                #print('out2_soft[:labeled_bs, 1, ...]',out2_soft[:labeled_bs, 1, ...].shape)
                #print('label_batch[:labeled_bs, ...]',label_batch[:labeled_bs, ...].shape)

                # deep supervision #深度监督损失
                #los1 = losses.dice_loss(out1_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                los2 = losses.dice_loss(out2_soft[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)
                los3 = losses.dice_loss(out3_soft[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)
                los4 = losses.dice_loss(out4_soft[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)
                los5 = losses.dice_loss(out5_soft[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)
                los = 0.6 * los2 + 0.4 * los3 + 0.2 * los4 + 0.1 * los5
                    # loss_seg += F.cross_entropy(y_prob[:labeled_bs], label_batch[:labeled_bs])
                    #loss_seg_dice += dice_loss(y_prob[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)
                    # print("train__-140_",y_prob[:labeled_bs, 1, ...].shape)
                    # print("train_____141__",label_batch[:labeled_bs, ...].shape)

                y_all[0] = y_prob1
                y_all[1] = y_prob2
                
                loss_seg_dice=loss_sup

                loss_consist = 0
                loss_conflict=0
                pixel_consist = 0
                # _, pseudo_outputs_1 = torch.max(y_all[0], dim=1) #pseudo_outputs_1每个位置的预测类别标签。
                # _, pseudo_outputs_2 = torch.max(y_all[1],dim=1)
                # mtx_bool_conflict = pseudo_outputs_1 != pseudo_outputs_2 #检查两个模型的预测类别是否相同，相同为1，不同为0
                # #mtx_bool_conflict = pseudo_outputs_1 != pseudo_outputs_2
                # conflict_ratio = mtx_bool_conflict.float().sum() / (args.max_samples-args.labelnum) #衡量了两个预测之间的不一致程度，即两个模型在同一像素上的类别预测是否发生冲突。

                # # entropy，熵是衡量概率分布不确定性的指标，在这里它用于表示每个像素位置上的预测分布的不确定性。，
                # entropy_1 = -torch.sum(y_all[0]* torch.log2(y_all[0] + 1e-10), dim=1) #加了负号，则越大确定性越大
                # entropy_2 = -torch.sum(y_all[1] * torch.log2(y_all[1]+ 1e-10), dim=1)

                # # weighted sum， 熵越小，表示模型对该像素位置的预测越自信，因此赋予更高的权重。
                # weights_1 = torch.exp(-entropy_1) / (torch.exp(-entropy_1) + torch.exp(-entropy_2))
                # weights_2 = 1 - weights_1
                # weighted_outputs = weights_1.unsqueeze(1) * y_all[0] + weights_2.unsqueeze(
                #     1) * y_all[1] #加权输出

                # weighted_outputs = torch.pow(weighted_outputs, 1.0 / 1) 
                # weighted_outputs = weighted_outputs / torch.sum(weighted_outputs, dim=1, keepdim=True)  #归一化

                # # get final outputs
                # pseudo_logits, pseudo_outputs = torch.max(weighted_outputs, dim=1)#从加权后的预测中选择最大值对应的类别

                for i in range(num_outputs):
                    for j in range(num_outputs):
                        if i != j:
                            uncertainty_o1 = -1.0 * torch.sum(y_all[i] * torch.log(y_all[i] + 1e-6), dim=1)
                            uncertainty_o2 = -1.0 * torch.sum(y_all[j] * torch.log(y_all[j] + 1e-6), dim=1)
                            mask = (uncertainty_o1 > uncertainty_o2).float()   #不确定性高的
                            mask1=(uncertainty_o1 <uncertainty_o2).float() #预测的都是不确定性低的，都是更加确定的
                            # print("2222222",y_all[j].shape)

                            batch_o, c_o, w_o, h_o, d_o = y_all[j].shape##当i=0,j=1,y_all[1]是第二个模型的预测结果
                            # print("1111111111",outfeats[j].shape)
                            batch_f, c_f, w_f, h_f, d_f = outfeats[j].shape #第二个人模型产生的特征原型图
                            teacher_o = y_all[j].reshape(batch_o, c_o, -1) #概率分布图，-1 表示自动计算维度，使得 batch_o * c_o * (H * W) 的元素总数保持不变。
                            teacher_f = outfeats[j].reshape(batch_f, c_f, -1)#特征图
                            stu_f = outfeats[i].reshape(batch_f, c_f, -1)
                            #计算教师模型的类原型

                            index = torch.argmax(y_all[j], dim=1, keepdim=True)#计算每个像素的最大类别概率。dim=1: 在类别维度上进行操作，即找出每个像素的预测类别。
                            prototype_bank = torch.zeros(batch_f, num_classes, c_f).cuda()#初始化零张量，batch_f 是批次大小，num_classes 是类别数量，c_f 是特征图的通道数。#用于存储每个类别的原型特征，用于后续计算模型 i 和模型 j 之间的一致性损失。
                            for ba in range(batch_f):#遍历每个批次中的图像。
                                for n_class in range(num_classes):#遍历每个类别
                                    mask_temp = (index[ba] == n_class).float()#创建一个掩码，标记当前批次图像中属于类别 n_class 的像素，形状：(1, H, W,d)，类别为 n_class 的像素为 1，其他像素为 0
                                    top_fea = outfeats[j][ba] * mask_temp #将特征图与掩码相乘，提取属于类别 n_class 的像素的特征。
                                    prototype_bank[ba, n_class] = top_fea.sum(-1).sum(-1).sum(-1) / (mask_temp.sum() + 1e-6)##对特征图的所有空间维度进行求和，得到类别 n_class 的原型特征。
                        # 计算类原型和教师特征图的余弦相似性
                            prototype_bank = F.normalize(prototype_bank, dim=-1)#对 prototype_bank 的最后一个维度（特征维度 c_f）进行归一化处理。归一化原型特征向量，以便进行余弦相似性计算
                            mask_t = torch.zeros_like(y_all[i]).cuda()
                            for ba in range(batch_o):
                                for n_class in range(num_classes):
                                    class_prototype = prototype_bank[ba, n_class]#获取当前类别 n_class 的类别原型特征向量。
                                    mask_t[ba, n_class] = F.cosine_similarity(teacher_f[ba], #来自倒数第二层的特征
                                                                              class_prototype.unsqueeze(1),#类原型
                                                                              dim=0).view(w_f, h_f, d_f)
                            #mask_t是相关特征与类原型的相似性

                            weight_pixel_t = (1 - nn.MSELoss(reduction='none')(mask_t, y_all[j])).mean(1) #得到可靠性，越低，距离越大，说明可靠性就越低，这个值越大，越可靠
                            weight_pixel_t = weight_pixel_t * mask #
                            weight_pixel_t1=nn.MSELoss(reduction='none')(mask_t, y_all[j]).mean(1)#越大，伪标签越不可靠
                            weight_pixel_t1=weight_pixel_t1*mask1

                            loss_t = consistency_criterion(y_all[i], torch.argmax(y_all[j], dim=1).detach())
                            loss_consist += (loss_t * weight_pixel_t.detach()).sum() / (mask.sum() + 1e-6)
                            loss_conflict+=(loss_t * weight_pixel_t1.detach()).sum() / (mask.sum() + 1e-6)

                            # if var_param_conflict_weight > 1 or var_param_conflict_weight < 1:
                            #     thresh_mask = pseudo_logits.ge(args["conf_threshold"]).bool()
                            #     if var_param_conflict_weight > 0 or var_param_conflict_weight < 0:
                            #         target_conflct = pseudo_outputs.clone()
                            #         target_conflct[~mtx_bool_conflict] = -100
                            #         loss_ulb_conflict = F.cross_entropy(y_all[i], target_conflct.long(),
                            #                                             ignore_index=-100, reduction="mean")
                            #
                            #         loss_conflict +=(loss_ulb_conflict*weight_pixel_t.detach()).sum() / (mask.sum() + 1e-6)

                #loss_ulb = var_param_conflict_weight * loss_conflict


                consistency_weight = get_current_consistency_weight(iter_num // 150)

                #loss = args.lamda * loss_seg_dice + consistency_weight * (loss_consist) +args.lamda*los+consistency_weight*loss_conflict
                optimizer.zero_grad()
                loss_l=args.lamda * loss_seg_dice+args.lamda * los
                loss_seg_dice.backward(retain_graph=True)

                # Grad-ReLU策略：在分类层更新参数时，将无监督损失的梯度设置为零
                for param in model.conv8.parameters():
                    if param.grad is not None:
                        param.grad = param.grad * torch.gt(param.data, 0).float()
                for param in model.conv8_2.parameters():
                    if param.grad is not None:
                        param.grad = param.grad * torch.gt(param.data, 0).float()




                # 更新特征提取层参数（包括有监督损失和无监督损失的梯度）
                optimizer.zero_grad()
                loss = args.lamda * loss_seg_dice + consistency_weight * (loss_consist) +args.lamda*los+consistency_weight*loss_conflict

                loss.backward()
                optimizer.step()
            iter_num = iter_num + 1
            logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f,loss_ds:%03f,loss_ub:%03f' %
                         (iter_num, loss, loss_seg_dice, loss_consist,los,loss_conflict))

            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice, iter_num)
            # writer.add_scalar('Labeled_loss/loss_seg_ce', loss_seg, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)

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
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
            iterator.close()
            break

    writer.close()
