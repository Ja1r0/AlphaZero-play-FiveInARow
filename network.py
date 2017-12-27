'''
input:
======
the current situation with history situations of the game board
form:
-----


'''
import torch
import torch.nn as nn
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, padding=0, bias=False)

class Input_block(nn.Module):
    def __init__(self,in_planes=17):
        super(Input_block,self).__init__()
        self.conv=conv3x3(in_planes,256)
        self.bn=nn.BatchNorm2d(256)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        out=self.relu(x)
        return out

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
    def __init__(self, in_planes=256, out_planes=256):
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
        return out


class Policy_head(nn.Module):
    def __init__(self, in_planes=256, in_size=7):
        super(Policy_head, self).__init__()
        self.conv = conv1x1(in_planes, out_planes=2)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_size * in_size * 2, in_size*in_size+1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        P = self.fc(out)
        return P


class Value_head(nn.Module):
    def __init__(self, in_planes=256,in_size=7):
        super(Value_head, self).__init__()
        self.conv = conv1x1(in_planes, out_planes=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1=nn.Linear(in_size*in_size*1,256)
        self.relu2=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(256,1)
        self.tanh=nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu1(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.relu2(x)
        x=self.fc2(x)
        v=self.tanh(x) # output a scalar in the range [-1,1]
        return v


class Network(nn.Module):
    def __init__(self,block_num,bord_size):
        super(Network,self).__init__()
        self.input_block=Input_block()
        resblocks=[]
        res_block=Res_block()
        for i in range(block_num):
            resblocks.append(res_block)
        self.res_tower=nn.Sequential(*resblocks)
        self.value_head=Value_head(in_size=bord_size)
        self.policy_head=Policy_head(in_size=bord_size)
    def forward(self, s):
        '''
        Input
        =====
        s:{Variable} size Nx17x7x7
        Output
        ======
        P:{Variable} size Nx(7x7+1)
        v:{Variable} size Nx1 in the range [-1,1]
        '''
        x=self.input_block.forward(s)
        x=self.res_tower(x)
        P=self.policy_head.forward(x)
        v=self.value_head.forward(x)
        return P, v
    def update(self, samples):
        pass

if __name__=='__main__':
    s=torch.randn(32,17,7,7)
    s=Variable(s.cuda())
    net=Network(19,7)
    net=net.cuda()
    P,v=net(s)
    print('P=',P,'\n','v=',v)