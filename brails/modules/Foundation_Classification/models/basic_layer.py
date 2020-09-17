import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.bn1 = nn.BatchNorm1d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channel, int(out_channel/4), 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(int(out_channel/4))
        self.conv2 = nn.Conv1d(int(out_channel / 4), int(out_channel / 4), 3, stride=stride, padding = 1, bias=False)
        self.bn3 = nn.BatchNorm1d(int(out_channel/4))
        self.conv3 = nn.Conv1d(int(out_channel / 4), out_channel, 1, stride=1, bias=False)
        self.conv4 = nn.Conv1d(in_channel, out_channel, 1, stride=stride, bias=False)
    def forward(self, x):
        res = x
        x = self.bn1(x)
        out_1 = self.relu(x)
        x = out_1
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        if(self.in_channel != self.out_channel) or (self.stride!=1):
            res = self.conv4(out_1)
        x += res
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, int(out_channel/4), 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(out_channel/4))
        self.conv2 = nn.Conv2d(int(out_channel / 4), int(out_channel / 4), 3, stride=stride, padding = 1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_channel/4))
        self.conv3 = nn.Conv2d(int(out_channel / 4), out_channel, 1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False)
    def forward(self, x):
        res = x
        x = self.bn1(x)
        out_1 = self.relu(x)
        x = out_1
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        if(self.in_channel != self.out_channel) or (self.stride!=1):
            res = self.conv4(out_1)
        x += res
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, int(out_channel/4), 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(out_channel/4))
        self.conv2 = nn.Conv2d(int(out_channel / 4), int(out_channel / 4), 3, stride=stride, padding = 1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_channel/4))
        self.conv3 = nn.Conv2d(int(out_channel / 4), out_channel, 1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False)
    def forward(self, x):
        res = x
        x = self.bn1(x)
        out_1 = self.relu(x)
        x = out_1
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        if(self.in_channel != self.out_channel) or (self.stride!=1):
            res = self.conv4(out_1)
        x += res
        return x

