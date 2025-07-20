import torch
from torch import nn
from torchsummary import summary 

class ConvBlock(nn.Module): #自定义的卷积神经网络模块
#n_stages：卷积块中的卷积层数量。n_filters_in：输入通道数。n_filters_out：输出通道数。normalization：归一化层类型，可以是 'none'、'batchnorm'、'groupnorm' 或 'instancenorm'。
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__() #调用父类 nn.Module 的初始化方法。
 
        ops = []  #创建一个空列表 ops，用来存储各层操作
        for i in range(n_stages):  #对于第一个卷积层，使用输入通道 n_filters_in，其他层使用输出通道 n_filters_out。
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1)) #添加 3D 卷积层，卷积核大小为 3x3x3，padding 为 1 保持输入和输出的空间尺寸相同。
            if normalization == 'batchnorm': #添加归一化层
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True)) #添加激活函数

        self.conv = nn.Sequential(*ops) #所有层打包成一个 Sequential 容器，使得这些层按照顺序依次执行。

    def forward(self, x):
        x = self.conv(x)
        return x

#残差学习（Residual Learning）机制的卷积块。
class ResidualConvBlock(nn.Module):

    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x) #: 残差连接。将卷积块的输出与输入进行逐元素相加。这是一种跳跃连接技术，帮助训练更深的网络并缓解梯度消失问题。
        x = self.relu(x)
        return x

#下采样的卷积块。用于在卷积神经网络中进行特征图的尺寸缩小
class DownsamplingConvBlock(nn.Module):

    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):#积操作的步长，用于控制下采样的程度。默认值为 2，表示每次步长为 2。
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))#改变特征图的通道数。这是通过卷积层的 n_filters_in 和 n_filters_out 参数来实现的。

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

#征图上采样的模块，在卷积神经网络中用于扩大特征图的空间维度
class Upsampling_function(nn.Module):
#stride上采样的比例因子。默认为 2，表示将特征图的每个空间维度大小扩大为原来的 2 倍
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))#nn.ConvTranspose3d: 反卷积层（或转置卷积），用来进行上采样。stride 决定了上采样的比例，padding=0 保持了输出空间维度的调整。
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))#通过 trilinear 上采样进行空间维度扩展。
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))#紧接着应用一个 3D 卷积层，kernel_size=3 和 padding=1 保持特征图的空间维度。conv3d对尺寸有影响吗

        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))#通过 nearest 上采样进行空间维度扩展。
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

#编码器部分。它利用卷积层、下采样层、激活函数等来提取特征
class Encoder(nn.Module):

    def __init__(self,
                 n_channels=3, #输入图像的通道数，通常为 3（RGB 图像）或 1（灰度图像）
                 n_classes=2, #分类任务中的类别数目，但在这个 Encoder 类中并没有直接用到
                 n_filters=16, #第一个卷积层的输出通道数，也是后续卷积层输出通道数的基数。重要！！！！！！！！
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
#self.block_one 到 self.block_five：五个卷积块，每个卷积块由一系列卷积层、归一化层和激活函数组成。
#self.block_one_dw 到 self.block_four_dw：每个卷积块后跟随的下采样层。
        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)#1 表示该卷积块有一个卷积层。
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5] #其中包含五个特征图 x1, x2, x3, x4, 和 x5，这些特征图可以用于后续的解码器部分进行跳跃连接（skip connections）。
        return res


class Decoder(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False,
                 up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16,
                                                 n_filters * 8,
                                                 normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8,
                                                n_filters * 4,
                                                normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4,
                                                  n_filters * 2,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2,
                                                  n_filters,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg


class Decoder_v1(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False,
                 up_type=0):#up_type: 上采样的模式，0 是转置卷积，1 是双线性插值，2 是最近邻插值。
        super(Decoder_v1, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock#has_residual 为 True，则使用 ResidualConvBlock 代替 ConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16,#输入通道数
                                                 n_filters * 8,#用于将特征图的分辨率上采样到 n_filters * 8。输出通道数
                                                 normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)#用于处理上采样后的特征图，使用 3 层卷积层。
        self.block_six_up = Upsampling_function(n_filters * 8,
                                                n_filters * 4,
                                                normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4,
                                                  n_filters * 2,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2,
                                                  n_filters,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)#ConvBlock 用于处理上采样后的特征图，使用 1 层卷积层。
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0) #最后的 1x1x1 卷积层用于将特征图映射到类别数 n_classes。

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):#features: 从编码器传递来的特征图列表，包含从低到高的特征图。
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)#x5_up: 使用 block_five_up 对 x5 进行上采样，。
        x5_up = x5_up + x4#并将其与 x4 进行跳跃连接

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)#通过 block_nine 处理最终的特征图。
        if self.has_dropout:
            x9 = self.dropout(x9)  #通过 block_nine 处理最终的特征图。
        out_seg = self.out_conv(x9)

        return out_seg, x9


class Decoder_v2(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False,
                 up_type=0):
        super(Decoder_v2, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16,
                                                 n_filters * 8,
                                                 normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8,
                                                n_filters * 4,
                                                normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4,
                                                  n_filters * 2,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2,
                                                  n_filters,
                                                  normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        import ipdb
        ipdb.set_trace()
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg, x9


class VNet(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1

class VNet_v1(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(VNet_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        if self.training:
            return out_seg1, f1
        return out_seg1



class MCNet3d_v1(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(MCNet3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2


class MCNet3d_v2(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(MCNet3d_v2, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        out_seg3 = self.decoder3(features)
        return out_seg1, out_seg2, out_seg3


class Mine3d_v1(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(Mine3d_v1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        out_seg2, f2 = self.decoder2(features)
        if self.training:
            return [out_seg1, out_seg2], [f1, f2]
        else:
            return out_seg1, out_seg2


class Mine3d_v1_drop(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(Mine3d_v1_drop, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def Dropout(self, x, p=0.3):
        x = torch.nn.functional.dropout3d(x, p)
        return x

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        # import ipdb; ipdb.set_trace()
        features_drop = [self.Dropout(i) for i in features]
        out_seg2, f2 = self.decoder2(features_drop)
        if self.training:
            return [out_seg1, out_seg2], [f1, f2]
        else:
            return out_seg1, out_seg2


class Mine3d_v1_drop_02(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(Mine3d_v1_drop_02, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

    def Dropout(self, x, p=0.3):
        x = torch.nn.functional.dropout3d(x, p)
        return x

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        # import ipdb; ipdb.set_trace()
        features_drop = [self.Dropout(i) for i in features]
        out_seg2, f2 = self.decoder2(features_drop)
        if self.training:
            return [out_seg1, out_seg2], [f1, f2]
        else:
            return out_seg1, out_seg2


class Mine3d_v1_pro(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(Mine3d_v1_pro, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

        self.projector1 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=1), nn.PReLU(), nn.Conv3d(16, 32, kernel_size=1))
        self.projector2 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=1), nn.PReLU(), nn.Conv3d(16, 32, kernel_size=1))

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        out_seg2, f2 = self.decoder2(features)
        f1 = self.projector1(f1)
        f2 = self.projector2(f2)
        if self.training:
            return [out_seg1, out_seg2], [f1, f2]
        else:
            return out_seg1, out_seg2


class Mine3d_v2(nn.Module):

    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(Mine3d_v2, self).__init__()

        # add project head for infonce loss
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder_v1(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1, f1 = self.decoder1(features)
        out_seg2, f2 = self.decoder2(features)
        import ipdb
        ipdb.set_trace()
        if self.training:
            return out_seg1, out_seg2, f1, f2
        else:
            return out_seg1, out_seg2



if __name__ == "__main__":
    model = Mine3d_v1().cuda()
    # model = nnMambaEncoder().cuda()

    input = torch.zeros((4, 3, 128, 128, 80)).cuda() #一个包含 8 个样本的批次，每个样本是一个 128x128 像素的单通道 3D 数据，深度为 80。
    summary(model, input.shape[1:],col_names=["input_size", "output_size", "num_params"])
    output = model(input)
    print(output.shape)