import torch
import torch.nn.functional as F
from functools import partial


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * (torch.tanh(F.softplus(x)))


def scaled_tanh(x, a=0.5):
    return a * torch.tanh(x)


def scaled_sigmoid(x, scale=2.0):
    return scale * torch.sigmoid(x)


def crelu(x, inplace=False):
    return torch.clamp(x, min=0.0, max=1.0)


activation = {'sigmoid': torch.sigmoid,
              'ssigmoid': partial(scaled_sigmoid, scale=0.5),
              'tanh': torch.tanh,
              'stanh': partial(scaled_tanh, a=0.5),
              'relu': F.relu,
              'relu6': F.relu6,
              'lrelu': F.leaky_relu,
              'nlrelu': partial(F.leaky_relu, negative_slope=-0.1),
              'crelu': crelu,
              'swish': swish,
              'mish': mish,
}

act_func_list = [af_name for af_name in activation]
