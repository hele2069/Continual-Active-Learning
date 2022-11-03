import torch.nn as nn
import torch.nn.functional as F
import torch
import IPython
from resnet import ResNet18
from lenet import LeNet5
from copy import deepcopy

def get_net(name):
    if name == 'MNIST':
        return MLP
    elif name == 'FashionMNIST':
        return LeNet5
    elif name == 'SVHN':
        return K_CNN
    elif name == 'CIFAR10':
        return VGG # K-CNN, Net3
        # return ResNet18
    elif name == 'CIFAR10_WAAL':
        return VGG_feat, VGG_10_clf, VGG_10_dis

class VGG_10_fea(nn.Module):

    def __init__(self):

        super(VGG_10_fea, self).__init__()
        # the vgg model can be changed to vgg11/vgg16
        # vgg 11 for svhn
        # vgg 16 for cifar 10 and cifar 100
        self.features = self._make_layers(cfg['VGG16'])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg):

        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_feat(nn.Module):
    def __init__(self, vgg_name, input_channels=3):
        super(VGG_feat, self).__init__()
        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        self.input_channels = input_channels
        self.features = self._make_layers(self.cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.classifier = nn.Sequential(nn.Linear(512, 50), 
                                         nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(50, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_10_clf(nn.Module):

    def __init__(self):
        super(VGG_10_clf, self).__init__()
        self.fc1 = nn.Linear(512,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class VGG_10_dis(nn.Module):
    def __init__(self):
        super(VGG_10_dis,self).__init__()
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self,x):
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x

    def get_embedding_dim(self):
        return 50

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x

    def get_embedding_dim(self):
        return 50

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x

    def get_embedding_dim(self):
        return 50

class Net_Correct(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(Net_Correct, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),            
        )
        # input_size = self._get_conv_output_size(input_shape)
        self.dense = nn.Sequential(nn.Linear(3872, 256))
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(self.dense(x))
        return x


class K_CNN(nn.Module):
    def __init__(self):
        super(K_CNN, self).__init__()
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = self.conv(x)
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = self.conv(x)
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        return e1

    def get_embedding_dim(self):
        return 50

class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16', input_channels=3):
        super(VGG, self).__init__()

        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        self.input_channels = input_channels
        self.features = self._make_layers(self.cfg[vgg_name]) # 
        self.classifier = nn.Linear(512, 10)
        self.classifier = nn.Sequential(nn.Linear(512, 50), 
                                         nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(50, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        # return F.softmax(out,dim=1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.feature = nn.Linear(28 * 28, 32)
        self.classifier = nn.Linear(32, 10)
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.feature(x)
        x = F.torch.relu(x)
        return self.classifier(x)
    
    def get_embedding(self, x):
        x = x.flatten(start_dim=1)
        x = self.feature(x)
        x = F.torch.relu(x)
        return x
    
    def get_embedding_dim(self):
        return 32
