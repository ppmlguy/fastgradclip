import torch
import torch.nn as nn
import torch.nn.functional as F
from fastgc.model.penet import PeGradNet
from fastgc.layers.linear import Linear
from fastgc.layers.conv import Conv2d
from fastgc.util import conv_outsize
from fastgc.activation import activation


class CNN(PeGradNet):
    def __init__(self, input_size, channel_sizes, kernel_sizes, fc_sizes,
                 num_classes, train_alg='batch'):
        super(type(self), self).__init__()

        self.input_size = input_size
        self.kernel_sizes = kernel_sizes
        self.act = F.relu
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # convolutional layers
        layers = []
        out_size = input_size
        for c_in, c_out, k in zip(channel_sizes[:-1], channel_sizes[1:], kernel_sizes):
            layer = Conv2d(c_in, c_out, k)
            layers.append(layer)
            out_size = conv_outsize(out_size, k, layer.padding[0], layer.stride[0])
            out_size = conv_outsize(out_size, self.pooling.kernel_size,
                                    self.pooling.padding, self.pooling.stride)

        self.convs = nn.ModuleList(layers)
        self.conv_outsize = out_size * out_size * c_out

        # fully-connected layers
        fc_sizes = [self.conv_outsize] + fc_sizes
        self.linears = nn.ModuleList([Linear(in_size, out_size)
                                      for in_size, out_size in zip(fc_sizes[:-1],
                                                                   fc_sizes[1:])])
        self.output_layer = Linear(fc_sizes[-1], num_classes)

        self.layers = [layer for layer in self.convs]
        self.layers += [layer for layer in self.linears]
        self.layers.append(self.output_layer)
        self.train_alg = train_alg

    def forward(self, x):
        out = x

        # convolutional layers
        for layer in self.convs:
            out = self.pooling(self.act(layer(out)))

        # flatten
        out = out.view(-1, self.conv_outsize)

        for layer in self.linears:
            out = self.act(layer(out))

        logits = self.output_layer(out)

        return logits
