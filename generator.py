# -*- coding: utf-8 -*-
""

@author: Qing, Modified by Rushdi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad

class generatorNew(nn.Module):
    # initializers
    def __init__(self,params, out_channel=2):
        super().__init__()
        factor = params['factor']
        siz_latent = params['siz_l']
        d = params['gen_base_size']
        self.gen_reg = params['gen_reg']
        
        if isinstance(params['slice'], int):
            self.nsl = 1
        else:
            self.nsl = len(params['slice'])
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(siz_latent, 100, 1, 1, 0),
            #nn.BatchNorm2d(100),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Tanh(), # Finish Layer 1 --> 1x1
            nn.Conv2d(100, d*8, 3, 1, 2),
            #nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 2 --> 3x3
            nn.Conv2d(d*8, d*8, 3, 1, 2),
            #nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 3 --> 5x5
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 10x10
            nn.Conv2d(d*8, d*4, 3, 1, 1),
            #nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 4 --> 10x10
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 20x20
            nn.Conv2d(d*4, d*4, 3, 1, 1),
            #nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 5 --> 20x20
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 40x40
            nn.Conv2d(d*4, d*2, 3, 1, 2),
            #nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 6 --> 42x42
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 84x84
            nn.Conv2d(d*2, d, 3, 1, 1),
            #nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 7 --> 84x84
            nn.Upsample(scale_factor=2, mode='nearest'), # NN --> 168x168
            nn.Conv2d(d, d, 3, 1, 1),
            #nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 8 --> 168x168
            nn.Conv2d(d, out_channel*self.nsl, 3, 1, 1),
            #nn.BatchNorm2d(out_channel),
            nn.Tanh(), # Finish Layer 9 --> 168x168
            )
        
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Created generator with ",pytorch_total_params," parameters")
        print('It is %.2f times the image size' %((pytorch_total_params/168/168)))
        

    def weight_init(self):
        for layer in self._modules:
            if layer.find('Conv') != -1:
                layer.weight.data.normal_(0.0, 0.02)
            elif layer.find('BatchNorm2d') != -1:
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)


    # forward method
    def forward(self, input):
        x = self.conv_blocks(input)
        x = x[:,0:self.nsl]+1j*x[:,self.nsl:2*self.nsl]
        x = x.permute((0,2,3,1))
        x = x.unsqueeze(1)
        return x

#%%        
    def weightl1norm(self):
        L1norm = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                L1norm = L1norm + torch.norm(param, 1)
        return(self.gen_reg*L1norm)
