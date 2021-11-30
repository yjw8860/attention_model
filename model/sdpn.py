import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import cv2

class DepthWiseUnit(nn.Module):
    def __init__(self,channels:int, repeat_num: int, dropout:float):
        super(DepthWiseUnit, self).__init__()
        self.repeat_num = repeat_num
        self.channels = channels
        self.dropout = dropout
        self.block = nn.Sequential(OrderedDict(self.depthwise()))
        self.norm_act = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(channels)),
            ("1_activation", nn.ReLU()),
        ]))

    def depthwise(self):
        output = []
        for i in range(self.repeat_num):
            output.append((f"{i}_0_depthwise_conv", nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels, bias=False)))
            output.append((f"{i}_1_normalization", nn.BatchNorm2d(self.channels)))
            output.append((f"{i}_2_activation", nn.ReLU()))
            output.append((f"{i}_3_dropout", nn.Dropout(self.dropout)))
            output.append((f"{i}_4_depthwise_conv", nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels, bias=False)))
            output.append((f"{i}_5_normalization", nn.BatchNorm2d(self.channels)))
            output.append((f"{i}_6_activation", nn.ReLU()))
            output.append((f"{i}_7_dropout", nn.Dropout(self.dropout)))
        return output

    def forward(self, x):
        x = self.norm_act(x)
        out_1 = self.block(x)
        out_2 = self.block(out_1)
        out_2 = out_1 + out_2
        return out_2

class PointWiseUnit(nn.Module):
    def __init__(self, in_channels:int, out_channels: int, dropout: float):
        super(PointWiseUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.block = nn.Sequential(OrderedDict(self.pointwise()))

    def pointwise(self):
        output = []
        output.append(("0_pointwise", nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)))
        output.append(("1_normalization", nn.BatchNorm2d(self.out_channels)))
        output.append(("2_activation", nn.ReLU()))
        output.append(("3_dropout", nn.Dropout(self.dropout)))
        return output

    def forward(self,x):
        return self.block(x)

class ClassifyUnit(nn.Module):
    def __init__(self, channels: int, dropout:float, class_nums: int):
        super(ClassifyUnit, self).__init__()
        self.channels = channels
        self.dropout = dropout
        self.class_nums = class_nums
        self.block_1 = nn.Sequential(OrderedDict(self.classify()))

    def classify(self):
        output = [
            ("0_downsampling", nn.Conv2d(self.channels,self.channels, kernel_size=3, padding=1, stride=2,bias=False)),
            ("1_normalization", nn.BatchNorm2d(self.channels)),
            ("2_activation", nn.ReLU()),
            ("3_dropout", nn.Dropout(self.dropout)),
            ("4_downsampling", nn.Conv2d(self.channels,self.channels, kernel_size=3, padding=1, stride=2, bias=False)),
            ("5_normalization", nn.BatchNorm2d(self.channels)),
            ("6_activation", nn.ReLU()),
            ("7_dropout", nn.Dropout(self.dropout)),
            ("8_downsampling", nn.Conv2d(self.channels,self.channels, kernel_size=3, padding=1, stride=2, bias=False)),
            ("9_normalization", nn.BatchNorm2d(self.channels)),
            ("10_activation", nn.ReLU()),
            ("11_dropout", nn.Dropout(self.dropout)),
            ("12_avgpooling", nn.AvgPool2d(kernel_size=2)),
            ("13_flatten", nn.Flatten()),
            ("14_classification", nn.Linear(in_features=2048, out_features=self.class_nums))
        ]
        return output

    def forward(self, x):
        return self.block_1(x)

class DepthwiseSeparableConvNet(nn.Module):
    def __init__(self):
        super(DepthwiseSeparableConvNet, self).__init__()
        self.out = nn.Sequential(OrderedDict([
            ("standard_convolution", nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)),
            ("depthwise_block_1", DepthWiseUnit(channels=64, repeat_num=1, dropout=0.5)),
            ("pointwise_block_1", PointWiseUnit(in_channels=64, out_channels=128, dropout=0.5)),
            ("depthwise_block_2", DepthWiseUnit(channels=128, repeat_num=1, dropout=0.5)),
            ("pointwise_block_2", PointWiseUnit(in_channels=128, out_channels=256, dropout=0.5)),
            ("depthwise_block_3", DepthWiseUnit(channels=256, repeat_num=1, dropout=0.5)),
            ("pointwise_block_3", PointWiseUnit(in_channels=256, out_channels=512, dropout=0.5)),
            ("flatten", ClassifyUnit(channels=512, dropout=0.5, class_nums=10))
        ]))
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self,x):
        return self.out(x)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class SDPNet(nn.Module):
    def __init__(self, img_size, class_length):
        super(SDPNet, self).__init__()
        self.img_size = img_size
        self.class_length = class_length
        self.feature = nn.Sequential(OrderedDict([
            ("standard_convolution_1", nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2)),
            ("depthwise_block_1", DepthWiseUnit(channels=64, repeat_num=1, dropout=0.5)),
            ("pointwise_block_1", PointWiseUnit(in_channels=64, out_channels=128, dropout=0.5)),
            ("depthwise_block_2", DepthWiseUnit(channels=128, repeat_num=1, dropout=0.5)),
            ("pointwise_block_2", PointWiseUnit(in_channels=128, out_channels=256, dropout=0.5)),
            ("depthwise_block_3", DepthWiseUnit(channels=256, repeat_num=1, dropout=0.5)),
            ("pointwise_block_3", PointWiseUnit(in_channels=256, out_channels=512, dropout=0.5)),
        ])) #output = (batch size, 512, 112, 112)
        self.pooling = nn.Sequential(OrderedDict([
            ("pooling_ReLU_0", nn.ReLU()),
            ("pooling_AdaptiveAvgPooling", nn.AdaptiveAvgPool2d(output_size = (1,1)))
        ]))
        self.basic_fc = nn.Sequential(OrderedDict([
            ("fc_Flatten", Flatten()),
            ('fc_Linear_1', nn.Linear(512, 128)),
            ('fc_batchnorm_1', nn.BatchNorm1d(num_features=128)),
            ('fc_ReLU_1', nn.ReLU()),
            ('fc_Linear_2',nn.Linear(128, self.class_length)),
            ('fc_batchnorm_2', nn.BatchNorm1d(num_features=self.class_length)),
            ('fc_Sigmoid', nn.Sigmoid())
        ]))
        self.fusion_fc = nn.Sequential(OrderedDict([
            ("fusion_Flatten", Flatten()),
            ("fusion_Linear_1", nn.Linear(512*2, 512)),
            ('fusion_batchnorm_1', nn.BatchNorm1d(num_features=512)),
            ("fusion_ReLU_1", nn.ReLU()),
            ("fusion_Linear_2", nn.Linear(512, 128)),
            ('fusion_batchnorm_2', nn.BatchNorm1d(num_features=128)),
            ("fusion_ReLU_2", nn.ReLU()),
            ("fusion_Linear_3", nn.Linear(128, self.class_length)),
            ('fusion_batchnorm_3', nn.BatchNorm1d(self.class_length)),
            ("fusion_Sigmoid", nn.Sigmoid())
        ]))

class SDPNetGlobal(nn.Module):
    def __init__(self, img_size, class_length):
        super(SDPNetGlobal, self).__init__()
        self.SDPN = SDPNet(img_size, class_length)

    def forward(self, x):
        f = self.SDPN.feature(x)
        p = self.SDPN.pooling(f)
        o = self.SDPN.basic_fc(p)

        return o, p, f

    def gOutput(self, x):
        outputs, poolings, features = self.forward(x)
        batch, _, height, width = x.size()
        attention = torch.max(features, 1).values.cpu().data.numpy()
        attentions = []
        for b in range(batch):
            output = cv2.resize(attention[b], (height,width))
            output = output - np.min(output)
            attentions.append(output / np.max(output))
        attentions = torch.from_numpy(np.array(attentions)).float().cuda()
        attentions = torch.reshape(attentions, (x.size()[0], 1, x.size()[2], x.size()[3]))
        return outputs, poolings, attentions * x

class SDPNetLocal(nn.Module):
    def __init__(self, img_size, class_length):
        super(SDPNetLocal, self).__init__()
        self.SDPN = SDPNet(img_size, class_length)

    def forward(self,x):
        f = self.SDPN.feature(x)
        p = self.SDPN.pooling(f)
        o = self.SDPN.basic_fc(p)
        return o, p

class SDPNetFusion(nn.Module):
    def __init__(self, img_size, class_length):
        super(SDPNetFusion, self).__init__()
        self.SDPN = SDPNet(img_size, class_length)

    def forward(self,x):
        return self.SDPN.fusion_fc(x)