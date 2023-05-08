
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):

        if self.bn:
            out = F.leaky_relu(self.bn1(self.conv1(x)))
        else:
            out = F.leaky_relu(self.conv1(x))

        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class ResBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, level, hideen1, num_class, bn=False, stride=1):
        super(ResBlock_s, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride > 1:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)

        if level == 1:
            self.fc1 = nn.Sequential(
                nn.Linear(4096, num_class),
            )
        if level ==2:
            self.fc1 = nn.Sequential(
                nn.Linear(8192, num_class),
            )
        if level ==3:
            self.fc1 = nn.Sequential(
                nn.Linear(8192, num_class),
            )
        if level ==4:
            self.fc1 = nn.Sequential(
                nn.Linear(4096, num_class),
            )

    def forward(self, x):
        if self.bn:
            out = F.leaky_relu(self.bn1(self.conv1(x)))
        else:
            out = F.leaky_relu(self.conv1(x))

        out = self.conv2(out)
        out += self.shortcut(x)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
    
def resnet4(input_shape, level, hideen1, num_classes):
    net = []

    net += [nn.Conv2d(input_shape[0], 16, 3, 1, 1)]
    net += [nn.BatchNorm2d(16)]
    net += [nn.LeakyReLU()]

    net += [ResBlock(16, 16)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock(16, 32, stride=2)]

    if level == 2:
        return nn.Sequential(*net)
    
    net += [ResBlock(32, 32)]

    if level == 3:
        return nn.Sequential(*net)

    net += [ResBlock_s(32, 64, level=level,  hideen1=hideen1, num_class=num_classes, stride=2)]

    if level <= 4:
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


def resnet3(input_shape, level, hideen1, num_classes):
    net = []

    net += [nn.Conv2d(input_shape[0], 16, 3, 1, 1)]
    net += [nn.BatchNorm2d(16)]
    net += [nn.LeakyReLU()]
    net += [ResBlock(16, 16)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock(16, 32, stride=2)]

    if level == 2:
        return nn.Sequential(*net)
    
    net += [ResBlock_s(32, 32, level=level,  hideen1=hideen1, num_class=num_classes)]

    if level == 3:
        return nn.Sequential(*net)

    net += [ResBlock_s(32, 64, level=level,  hideen1=hideen1, num_class=num_classes, stride=2)]

    if level <= 4:
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


def resnet2(input_shape, level, hideen1, num_classes):
    net = []

    net += [nn.Conv2d(input_shape[0], 16, 3, 1, 1)]
    net += [nn.LeakyReLU()]
    net += [ResBlock(16, 16)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock_s(16, 32, level=level,  hideen1=hideen1, num_class=num_classes,stride=2)]

    if level == 2:
        return nn.Sequential(*net)
    
    net += [ResBlock_s(32, 32, level=level,  hideen1=hideen1, num_class=num_classes)]

    if level == 3:
        return nn.Sequential(*net)

    net += [ResBlock_s(32, 64, level=level,  hideen1=hideen1, num_class=num_classes, stride=2)]

    if level <= 4:
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


def resnet1(input_shape, level, hideen1, num_classes):
    net = []

    net += [nn.Conv2d(input_shape[0], 16, 3, 1, 1)]
    net += [nn.LeakyReLU()]
    net += [ResBlock_s(16, 16, level=level,  hideen1=hideen1, num_class=num_classes, stride=2)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock_s(16, 32,  level=level,  hideen1=hideen1, num_class=num_classes, stride=2)]

    if level == 2:
        return nn.Sequential(*net)
    
    net += [ResBlock_s(32, 32, level=level,  hideen1=hideen1, num_class=num_classes)]

    if level == 3:
        return nn.Sequential(*net)

    net += [ResBlock_s(32, 64, level=level,  hideen1=hideen1, num_class=num_classes, stride=2)]

    if level <= 4:
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


def ResNet18(level, hideen1, num_classes):
    if level ==1:
        return resnet1(input_shape=(3,32,32), level=level, hideen1=hideen1, num_classes=num_classes)
    if level ==2:
        return resnet2(input_shape=(3,32,32), level=level, hideen1=hideen1, num_classes=num_classes)
    if level ==3:
        return resnet3(input_shape=(3,32,32), level=level, hideen1=hideen1, num_classes=num_classes)
    if level ==4:
        return resnet4(input_shape=(3,32,32), level=level, hideen1=hideen1, num_classes=num_classes)




class Server_ResNet(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=64, hideen2=128, hideen3=256, hideen4=128, hideen5=64, num_classes=2):
        super(Server_ResNet, self).__init__()
        act = nn.LeakyReLU
        self.fc2 = nn.Sequential(
            nn.Linear(hideen2, hideen3*3),
            act(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hideen3*3, hideen3*2),
            act(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hideen3*2, hideen3),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(hideen3, hideen4),
            act(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(hideen4, hideen5),
            act(),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(hideen5, num_classes),
        )


    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        out1 = self.fc2(x)
        out2 = self.fc3(out1)
        out3 = self.fc4(out2)
        out4 = self.fc5(out3)
        out5 = self.fc6(out4)
        out6 = self.fc7(out5)
        return out6




class Server_ResNet_3(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=64, hideen2=128, hideen3=256, hideen4=128, hideen5=64, num_classes=2):
        super(Server_ResNet_3, self).__init__()
        act = nn.LeakyReLU
        self.fc2 = nn.Sequential(
            nn.Linear(hideen2, hideen3*3),
            act(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hideen3*3, hideen3*2),
            act(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hideen3*2, hideen3),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(hideen3, hideen4),
            act(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(hideen4, hideen5),
            act(),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(hideen5, num_classes),
        )


    def forward(self, x1, x2, x3):
        x= torch.cat([x1, x2,x3], dim=1)
        out1 = self.fc2(x)
        out2 = self.fc3(out1)
        out3 = self.fc4(out2)
        out4 = self.fc5(out3)
        out5 = self.fc6(out4)
        out6 = self.fc7(out5)
        return out6



class Server_ResNet_4(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=64, hideen2=128, hideen3=256, hideen4=128, hideen5=64, num_classes=2):
        super(Server_ResNet_4, self).__init__()
        act = nn.LeakyReLU
        self.fc2 = nn.Sequential(
            nn.Linear(hideen2, hideen3*3),
            act(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hideen3*3, hideen3*2),
            act(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hideen3*2, hideen3),
            act(),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(hideen3, hideen4),
            act(),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(hideen4, hideen5),
            act(),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(hideen5, num_classes),
        )


    def forward(self, x1, x2, x3,x4):
        x= torch.cat([x1, x2,x3,x4], dim=1)
        out1 = self.fc2(x)
        out2 = self.fc3(out1)
        out3 = self.fc4(out2)
        out4 = self.fc5(out3)
        out5 = self.fc6(out4)
        out6 = self.fc7(out5)
        return out6



