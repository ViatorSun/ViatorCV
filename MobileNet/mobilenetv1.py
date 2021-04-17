#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time   :  2021.
# @Author :  绿色羽毛
# @Email  :  lvseyumao@foxmail.com
# @Blog   :  https://blog.csdn.net/ViatorSun
# @Note   :  MobileNetV1   创新：depthwise卷积





import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable






class MobileNet(nn.Module):
    def __init__(self, n_class=1000):
        super(MobileNet, self).__init__()
        self.nclass = n_class

        def conv_bn(inp, oup, stride):
            return nn.Sequential(   nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
                                    nn.BatchNorm2d(oup),
                                    nn.ReLU(inplace=True)   )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(   nn.Conv2d(inp, inp, kernel_size=(3,3), stride=stride, padding_mode='reflect', groups=inp, bias=False),  # depthwise卷积
                                    nn.BatchNorm2d(inp),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(inp, oup, kernel_size=(1,1), stride=(1,1),  padding_mode='zeros', bias=False),                # pointwise卷积
                                    nn.BatchNorm2d(oup),
                                    nn.ReLU(inplace=True)   )

        self.model = nn.Sequential( conv_bn(3  , 32 , 2),
                                    conv_dw(32 , 64 , 1),
                                    conv_dw(64 , 128, 2),
                                    conv_dw(128, 128, 1),
                                    conv_dw(128, 256, 2),
                                    conv_dw(256, 256, 1),
                                    conv_dw(256, 512, 2),
                                    conv_dw(512, 512, 1),
                                    conv_dw(512, 512, 1),
                                    conv_dw(512, 512, 1),
                                    conv_dw(512, 512, 1),
                                    conv_dw(512, 512, 1),
                                    conv_dw(512, 1024, 2),
                                    conv_dw(1024, 1024, 1),
                                    nn.AvgPool2d(7)         )
        self.fc = nn.Linear(1024, self.nclass)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x




def speed(model, name):
    # t0    = time.time()
    input = torch.rand(1,3,224,224).cuda()    # input = torch.rand(1,3,224,224).cuda()
    input = Variable(input, volatile = True)
    # t1    = time.time()

    model(input)

    t2 = time.time()
    for i in range(10):
        model(input)
    t3 = time.time()
    
    torch.save(model.state_dict(), "test_%s.pth"%name)
    print('%10s : %f' % (name, t3 - t2))





if __name__ == '__main__':
    #cudnn.benchmark = True # This will make network slow ??
    resnet18   = models.resnet18(num_classes=2).cuda()
    alexnet    = models.alexnet(num_classes=2).cuda()
    vgg16      = models.vgg16(num_classes=2).cuda()
    squeezenet = models.squeezenet1_0(num_classes=2).cuda()
    mobilenet  = MobileNet().cuda()
    # print(mobilenet)

    speed(resnet18  , 'resnet18')
    speed(alexnet   , 'alexnet')
    speed(vgg16     , 'vgg16')
    speed(squeezenet, 'squeezenet')
    speed(mobilenet , 'mobilenet')
