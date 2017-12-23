import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, padding=0, bias=False)


class Conv_block(nn.Module):
    def __init__(self, in_planes, out_planes=256):
        super(Conv_block, self).__init__()
        self.conv = conv3x3(in_planes, out_planes)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Res_block(nn.Module):
    def __init__(self, in_planes, out_planes=256):
        super(Res_block, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)


class Policy_head(nn.Module):
    def __init__(self, in_planes, in_size):
        super(Policy_head, self).__init__()
        self.conv = conv1x1(in_planes, out_planes=2)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_size * in_size * 2, 362)

    def forward(self, x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Value_head(nn.Module):
    def __init__(self, in_planes):
        super(Value_head, self).__init__()
        self.conv = conv1x1(in_planes, out_planes=1)
        self.bn = nn.BatchNorm2d(out_planes=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Network:
    def __init__(self):
        def predict(self, s, history):
            return P, v

    def update(self, samples):