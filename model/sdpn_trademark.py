from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

class DepthWiszeUnit(nn.Module):
    def __init__(self, channels: int, dropout_rate:float):
        super(DepthWiszeUnit, self).__init__()
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.batch_norm = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)
        self.depthwise_conv = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels, bias=False)

    def forward(self, x):
        out = F.dropout(self.depthwise_conv(self.relu(self.batch_norm(x))), p=self.dropout_rate, inplace=False, training=self.training)
