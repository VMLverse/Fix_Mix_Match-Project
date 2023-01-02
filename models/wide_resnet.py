# code in this file is adapted from
#  https://deepnote.com/@saurav-maheshkar/Wide-Residual-Networks-3a7a6e91-b4f6-45bb-84bd-e5b3c7b58fc8

import torch
import torch.nn as nn
import torch.nn.functional as F

#basic buildiing block for NetworkBlock
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        # layer set1
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        # layer set2
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate
        #if both in & out planes are equal
        self.equalInOut = (in_planes == out_planes)

        self.convShortcut = None
        # if inplanes & out planes are NOT equal, then set convShortcut to Conv2d
        if (not self.equalInOut):
            self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False)
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


#building block for each block layer in wide resnet
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            if (i==0): #if first layer
                stride_block = stride
                planes_in_block = in_planes
            else: # if not first layer ,then
                stride_block = 1
                planes_in_block = out_planes

            layers.append(block(planes_in_block, out_planes,
                                stride_block, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

#Wide ResNet implementation
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, dropout_rate=0.0, num_classes=10):
        super(WideResNet, self).__init__()

        # setup channels from 16, multiplying by widen factor
        thin_channel = 16
        #create an array of channels
        channels = {'thin_channel': thin_channel, 'wide_channel': (thin_channel * 1) * widen_factor, 'wider_channel': (thin_channel * 2) * widen_factor, 'widest_channel': (thin_channel * 4) * widen_factor}

        #make sure Wide-resnet depth is 6n+4
        assert ((depth - 4) % 6 == 0)
        #calculate n
        n = (depth-4)/6

        block = BasicBlock

        # 1st conv before any network block - conv3x3
        self.conv3x3 = nn.Conv2d(3, channels['thin_channel'], kernel_size=3, stride=1,padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, channels['thin_channel'], channels['wide_channel'], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, channels['wide_channel'], channels['wider_channel'], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, channels['wider_channel'], channels['widest_channel'], block, 2, dropout_rate)

        # Global Average Pooling and Classifier
        self.bn1 = nn.BatchNorm2d(channels['widest_channel'], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels['widest_channel'], num_classes)

        #initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

        self.channels = channels['widest_channel']

    def forward(self, x):
        out = self.conv3x3(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.fc(out)