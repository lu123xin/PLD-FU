import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from nnunet.network_architecture.neural_network import SegmentationNetwork

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module): #残差块
    expansion = 1 #块的输入和输出通道数是相同的
#inplanes: 输入通道数。planes:输出通道，stride: 步幅，downsample: 一个可选的层，可能是一个卷积层，mamba_layer: 一个可选的全局注意力层（Global Attention Layer），用于增强特征表示。
    def __init__(self, inplanes, planes, stride=1, downsample=None, mamba_layer=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride) #应用3x3卷积核，将输入通道inplanes转换为planes通道，并根据stride参数设置步幅。
        self.bn1 = nn.BatchNorm3d(planes)#第一个批量归一化层，对conv1的输出进行归一化。
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes) #第一个批量归一化层，对conv1的输出进行归一化。
        self.bn2 = nn.BatchNorm3d(planes) #第二个批量归一化层
        self.mamba_layer = mamba_layer #mamba层
        self.downsample = downsample #于调整残差连接的维度
        self.stride = stride #步幅

    def forward(self, x):
        identity = x #于调整残差连接的维度

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.mamba_layer is not None:
            global_att = self.mamba_layer(x) #计算全局注意力。
            out += global_att #将全局注意力结果加到当前的特征图上
        if self.downsample is not None:
            # if self.mamba_layer is not None:
            #     global_att = self.mamba_layer(x)
            #     identity = self.downsample(x+global_att)
            # else:
            identity = self.downsample(x) #将输入数据通过downsample层处理

        out += identity #加到特征图中
        out = self.relu(out)

        return out

#创建一个由多个 BasicBlock 组成的残差层
def make_res_layer(inplanes, planes, blocks, stride=1, mamba_layer=None):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    ) #创建 downsample 层

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes, mamba_layer=mamba_layer))

    return nn.Sequential(*layers) # 将所有的BasicBlock按顺序放入一个Sequential模块中，构建出一个完整的残差层。


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim #输入和输出的通道数
        self.nin = conv1x1(dim, dim)
        self.norm = nn.BatchNorm3d(dim) # LayerNorm
        self.relu = nn.ReLU(inplace=True)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model 
            d_state=d_state,  # SSM state expansion factor#Mamba模块中的状态扩展因子
            d_conv=d_conv,  # Local convolution width  Mamba模块中局部卷积的宽度
            expand=expand  # Block expansion factorMamba模块的扩展因子
        )
        
    def forward(self, x):
        B, C = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x) #批量归一化层。
        x = self.relu(x)
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2) #将输入展平为一个二维张量，以适应 Mamba 模块的输入
        # print('x_norm.dtype', x_norm.dtype)
        x_mamba = self.mamba(x_flat) #应用 Mamba 模块进行特征变换
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims) #将 Mamba 模块的输出还原为原始的 3D 张量形式。
        return out

#包含两个卷积层的模块
class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ) #创建一个合起来的模块

    def forward(self, input):
        return self.conv(input)

#简单而高效的神经网络层模块，用于在3D卷积网络中进行特征提取
class SingleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)

#自定义的注意力机制层，用于在神经网络中对特征进行加权
class Attentionlayer(nn.Module):
    def __init__(self,dim,r=16,act='relu'): #r维度缩减因子。默认值为16，用于控制注意力计算的中间层大小。
        super(Attentionlayer, self).__init__()
        self.layer1 = nn.Linear(dim, int(dim//r))
        self.layer2 = nn.Linear(int(dim//r), dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() #ayer2 的输出转换为 [0, 1] 范围的值，生成注意力权重。
    def forward(self, inp):#inp输入特征
        att = self.sigmoid(self.layer2(self.relu(self.layer1(inp))))
        return att.unsqueeze(-1) #在最后一个维度上增加一个新的维度


class nnMambaSeg(SegmentationNetwork):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=2):#in_ch: 输入通道数，通常为1（灰度图像）或3（RGB图像）。channels: 初始通道数，默认值为32，用于卷积操作。blocks: 每个残差块的数量，默认值为3。
        super(nnMambaSeg, self).__init__()
        self.do_ds = True
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3) #两个卷积层 + BatchNorm + ReLU，用于特征提取，初始卷积层。
        # self.mamba_layer_stem = MambaLayer(channels)
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))#自适应平均池化，将输入的特征图池化到固定的 (1, 1, 1) 尺寸。

        self.att1 = Attentionlayer(channels) #计算特征的注意力权重，用于调整不同层的特征图
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2, mamba_layer=MambaLayer(channels*2))
#建残差层，每层包括多个残差块和 MambaLayer，用于特征提取和处理
        self.att2 = Attentionlayer(channels*2)#计算特征的注意力权重，用于调整不同层的特征图
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2, mamba_layer=MambaLayer(channels*4))
        # self.mamba_layer_2 = MambaLayer(channels*4)

        self.att3 = Attentionlayer(channels*4)#计算特征的注意力权重，用于调整不同层的特征图
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2, mamba_layer=MambaLayer(channels*8))
        # self.mamba_layer_3 = MambaLayer(channels*8)
#Upsample: 上采样层，逐步将特征图的尺寸还原到输入图像的尺寸，DoubleConv: 对上采样后的特征图进行卷积操作，融合不同尺度的特征信息。
        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, number_classes)


        # self.ds1_cls_conv = nn.Conv3d(32, 4, kernel_size=1)
        # self.ds2_cls_conv = nn.Conv3d(64, 4, kernel_size=1)
        # self.ds3_cls_conv = nn.Conv3d(128, 4, kernel_size=1)
        self.ds1_cls_conv = nn.Conv3d(32, number_classes, kernel_size=1)#Conv3d: 用于生成不同层的分割预测结果。
        self.ds2_cls_conv = nn.Conv3d(64, number_classes, kernel_size=1)
        self.ds3_cls_conv = nn.Conv3d(128, number_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.in_conv(x) #对输入图像进行初始的卷积操作，得到 c1 特征图。
        scale_f1 = self.att1(self.pooling(c1).reshape(c1.shape[0], c1.shape[1])).reshape(c1.shape[0], c1.shape[1], 1, 1, 1)#算 c1 的注意力权重，并将其调整为相同的空间尺寸。
        # c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1) #经过第一个残差层进行特征提取
        # c2_s = self.mamba_layer_1(c2) + c2
        scale_f2 = self.att2(self.pooling(c2).reshape(c2.shape[0], c2.shape[1])).reshape(c2.shape[0], c2.shape[1], 1, 1, 1)
#计算 c2 的注意力权重，并将其调整为相同的空间尺寸
        c3 = self.layer2(c2)
        # c3_s = self.mamba_layer_2(c3) + c3
        scale_f3 = self.att3(self.pooling(c3).reshape(c3.shape[0], c3.shape[1])).reshape(c3.shape[0], c3.shape[1], 1, 1, 1)
        c4 = self.layer3(c3)
        # c4_s = self.mamba_layer_3(c4) + c4

        up_5 = self.up5(c4)#对 c4 进行上采样
        merge5 = torch.cat([up_5, c3*scale_f3], dim=1)#将上采样后的 c4 与经过注意力加权的 c3 拼接在一起。
        c5 = self.conv5(merge5) #对拼接后的特征图进行卷积操作。
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2*scale_f2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1*scale_f1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8) #对上采样后的特征图进行卷积操作，得到最终的分割结果。

        logits = [] #返回不同层的输出结果，用于多尺度预测
        logits.append(c8)
        logits.append(self.ds1_cls_conv(c7))
        logits.append(self.ds2_cls_conv(c6))
        logits.append(self.ds3_cls_conv(c5))
        # logits.append(self.ds3_cls_conv(c4))
        # print(np.shape(logits[0]),np.shape(logits[1]),np.shape(logits[2]),np.shape(logits[3]))
        if self.do_ds:
            return logits
        else:
            return logits[0]

        # return c8


class nnMambaSegSL(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=6):
        super(nnMambaSegSL, self).__init__()
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        # self.mamba_layer_stem = MambaLayer(channels)
        self.pooling =  nn.AdaptiveAvgPool3d((1, 1, 1))

        self.att1 = Attentionlayer(channels)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.mamba_layer_1 = MambaLayer(channels*2)

        self.att2 = Attentionlayer(channels*2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.mamba_layer_2 = MambaLayer(channels*4)

        self.att3 = Attentionlayer(channels*4)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.mamba_layer_3 = MambaLayer(channels*8)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels, number_classes)

    def forward(self, x):
        c1 = self.in_conv(x)
        scale_f1 = self.att1(self.pooling(c1).reshape(c1.shape[0], c1.shape[1])).reshape(c1.shape[0], c1.shape[1], 1, 1, 1)
        # c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1)
        # c2_s = self.mamba_layer_1(c2) + c2
        scale_f2 = self.att2(self.pooling(c2).reshape(c2.shape[0], c2.shape[1])).reshape(c2.shape[0], c2.shape[1], 1, 1, 1)

        c3 = self.layer2(c2)
        # c3_s = self.mamba_layer_2(c3) + c3
        scale_f3 = self.att3(self.pooling(c3).reshape(c3.shape[0], c3.shape[1])).reshape(c3.shape[0], c3.shape[1], 1, 1, 1)
        c4 = self.layer3(c3)
        # c4_s = self.mamba_layer_3(c4) + c4

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3*scale_f3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2*scale_f2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1*scale_f1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        c8 = self.conv8(up_8)
        return c8


# class nnMambaSeg(nn.Module):
#     def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=6):
#         super(nnMambaSeg, self).__init__()
#         self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
#         # self.mamba_layer_stem = MambaLayer(channels)

#         self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
#         self.mamba_layer_1 = MambaLayer(channels*2)

#         self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
#         self.mamba_layer_2 = MambaLayer(channels*4)

#         self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
#         self.mamba_layer_3 = MambaLayer(channels*8)

#         self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.conv5 = DoubleConv(channels * 12, channels * 4)
#         self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.conv6 = DoubleConv(channels * 6, channels * 2)
#         self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.conv7 = DoubleConv(channels * 3, channels)
#         self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.conv8 = DoubleConv(channels, number_classes)

#     def forward(self, x):
#         c1 = self.in_conv(x)
#         # c1_s = self.mamba_layer_stem(c1) + c1
#         c2 = self.layer1(c1)
#         c2_s = self.mamba_layer_1(c2) + c2
#         c3 = self.layer2(c2_s)
#         c3_s = self.mamba_layer_2(c3) + c3
#         c4 = self.layer3(c3_s)
#         c4_s = self.mamba_layer_3(c4) + c4

#         up_5 = self.up5(c4_s)
#         merge5 = torch.cat([up_5, c3_s], dim=1)
#         c5 = self.conv5(merge5)
#         up_6 = self.up6(c5)
#         merge6 = torch.cat([up_6, c2_s], dim=1)
#         c6 = self.conv6(merge6)
#         up_7 = self.up7(c6)
#         merge7 = torch.cat([up_7, c1], dim=1)
#         c7 = self.conv7(merge7)
#         up_8 = self.up8(c7)
#         c8 = self.conv8(up_8)
#         return c8


class nnMambaEncoder(nn.Module):
    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=1):
        super(nnMambaEncoder, self).__init__()
        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        # self.mamba_layer_stem = MambaLayer(channels)

        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.mamba_layer_1 = MambaLayer(channels*2)

        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.mamba_layer_2 = MambaLayer(channels*4)

        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.mamba_layer_3 = MambaLayer(channels*8)

        self.pooling =  nn.AdaptiveAvgPool3d((1, 1, 1))
        self.mlp = nn.Sequential(nn.Linear(channels*8, channels), nn.ReLU(), nn.Dropout(0.5), nn.Linear(channels, number_classes))

        # self.sig = nn.Sigmoid()


    def forward(self, x):
        c1 = self.in_conv(x)
        # c1_s = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1)
        # c2_s = self.mamba_layer_1(c2) + c2
        c3 = self.layer2(c2)
        c3_s = self.mamba_layer_2(c3) + c3
        c4 = self.layer3(c3)
        c4_s = self.mamba_layer_3(c4) + c4
        c5 = self.pooling(c4).view(c4.shape[0], -1)
        c5 = self.mlp(c5)
        # c5 = self.sig(c5)
        return c5


if __name__ == "__main__":
    #model = nnMambaSeg().cuda()
    model = nnMambaEncoder().cuda()

    input = torch.zeros((8, 1, 128, 128, 128)).cuda()
    output = model(input)
    print(output.shape)