import torch
import torch.nn as nn
import torch.nn.functional as F
from fastgc.model.penet import PeGradNet
from fastgc.layers.linear import Linear
from fastgc.activation import activation


class MLP(PeGradNet):
    def __init__(self, input_size, hidden_sizes, output_size, act_func='sigmoid',
                 train_alg='batch'):
        super(MLP, self).__init__()

        self.input_size = input_size
        layer_sizes = [input_size] + hidden_sizes
        self.linears = nn.ModuleList([Linear(in_size, out_size, bias=True)
                                      for in_size, out_size in zip(layer_sizes[:-1],
                                                                   layer_sizes[1:])])

        self.output_layer = Linear(hidden_sizes[-1], output_size, bias=True)
        self.act = activation[act_func]
        self.train_alg=train_alg

        # list of layers in the network
        self.layers = [layer for layer in self.linears]
        self.layers.append(self.output_layer)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x

        for layer in self.linears:
            out = self.act(layer(out))

        logits = self.output_layer(out)

        return logits

