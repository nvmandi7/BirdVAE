
# From https://github.com/dariocazzani/pytorch-AE/blob/master/architectures.py

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as  np


class CNN_Encoder(nn.Module):
    def __init__(self, input_shape, output_size):
        super(CNN_Encoder, self).__init__()

        self.input_shape = input_shape
        self.channel_mult = 16

        #convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,
                     out_channels=self.channel_mult*1,
                     kernel_size=4,
                     stride=1,
                     padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
            nn.BatchNorm2d(self.channel_mult*16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv)

        self.linear = nn.Sequential(
            nn.Linear(self.flat_fts, output_size),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(0.2),
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_shape)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x.view(-1, *self.input_shape))
        x = x.view(-1, self.flat_fts)
        return self.linear(x)

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, output_shape=(1, 28, 28)):
        super(CNN_Decoder, self).__init__()
        self.output_channels, self.output_height, self.output_width = output_shape
        self.input_size = embedding_size
        self.channel_mult = 16
        
        self.fc_output_size = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.fc_output_size),
            nn.BatchNorm1d(self.fc_output_size),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            # output is Z, going into a convolution
            nn.ConvTranspose2d(self.fc_output_size, self.channel_mult*4,
                                4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            # state size. self.channel_mult*32 x 4 x 4
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
                                3, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            # state size. self.channel_mult*16 x 7 x 7
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            # state size. self.channel_mult*8 x 14 x 14
            nn.ConvTranspose2d(self.channel_mult*1, self.output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. self.output_channels x 28 x 28
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x.view(-1, self.output_width*self.output_height)