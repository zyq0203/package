import torch
import torch.nn as nn
from network import Conv2d,SELayer,Dilated2
import numpy as np
from torch.autograd import Variable


class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    def __init__(self, bn=False):
        super(MCNN, self).__init__()

        self.branch1 = nn.Sequential(Conv2d(1, 64, 3, bn=bn),
                                     Conv2d(64, 64, 3,bn=bn),
                                     nn.MaxPool2d(2))

        self.branch2 = nn.Sequential(Conv2d(64, 128, 3, bn=bn),
                                     Conv2d(128, 128, 3,bn=bn),
                                     nn.MaxPool2d(2))
        
        self.branch3 = nn.Sequential(Conv2d(128, 256, 3,bn=bn),
                                     Conv2d(256, 256, 3,bn=bn),
                                     Conv2d(256, 256, 3,bn=bn),
                                     nn.MaxPool2d(2))
 
        self.branch4 = nn.Sequential(Conv2d(256, 512, 3,bn=bn),
                                     Conv2d(512, 512, 3,bn=bn),
                                     Conv2d(512, 512, 3,bn=bn),
                                     nn.MaxPool2d(2),
                                     nn.Upsample(scale_factor= 2, mode='nearest'))
        
        self.branch5 = nn.Sequential(Conv2d(768,512,1,bn=bn),
                                     Conv2d(512,512,3,bn=bn),
                                     Conv2d(512,512,3,bn=bn),
                                     nn.Upsample(scale_factor= 2, mode='nearest'))
        
        self.branch6 = nn.Sequential(Conv2d(640,512,1,bn=bn),
                                     Conv2d(512,512,3,bn=bn),
                                     Conv2d(512,512,3,bn=bn),
                                     nn.Upsample(scale_factor= 2, mode='nearest'))

        self.branch7 = nn.Sequential(Dilated2(576,512,3,2, bn=bn),
                                     Dilated2(512,256,3,2, bn=bn),
                                     Dilated2(256,128,3,2, bn=bn),
                                     Dilated2(128, 64,3,2, bn=bn),
                                     Conv2d(64,1,1,bn=bn))
        self.se1 = SELayer(768, 192)                             
        self.se2 = SELayer(640, 160)
        self.se3 = SELayer(576, 144)
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        x4 = self.branch4(x3) 
        a1 = x3.shape[2:]
        b1 = x4.shape[2:]
        if b1!=a1:
            x4 = x4.data
            x4 = np.array([np.array(x4[0,i,:,:].resize_(a1)) for i in range(x4.shape[1])])
            x4 = np.expand_dims(x4,axis=0)
            x4 = Variable(torch.from_numpy(x4)).cuda()
        x5 = torch.cat((x4,x3),1)
        x51 = self.se1(x5)
        x51+=x5
        x6 = self.branch5(x51)
        a2 = x2.shape[2:]
        b2 = x6.shape[2:]
        if b2!=a2:
            x6 = x6.data
            x6 = np.array([np.array(x6[0,i,:,:].resize_(a2)) for i in range(x6.shape[1])])
            x6 = np.expand_dims(x6,axis=0)
            x6 = Variable(torch.from_numpy(x6)).cuda()
        x7 = torch.cat((x6,x2),1)
        x8 = self.se2(x7)
        x8 += x7
        x9 = self.branch6(x8)
        a3 = x1.shape[2:]
        b3 = x9.shape[2:]
        if b3!=a3:
            x9 = x9.data
            x9 = np.array([np.array(x9[0,i,:,:].resize_(a3)) for i in range(x9.shape[1])])
            x9 = np.expand_dims(x9,axis=0)
            x9 = Variable(torch.from_numpy(x9)).cuda()
        x10 = torch.cat((x1,x9),1)
        x11 = self.se3(x10)
        x11 += x10
        x = self.branch7(x11)
        return x
