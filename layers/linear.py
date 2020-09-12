import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

        self.pre_activation = None
        self.layer_input = None

    def forward(self, input):        
        self.layer_input = input

        out = F.linear(input, self.weight, self.bias)
        self.pre_activation = out            
        
        return self.pre_activation

    def per_example_gradient(self, deriv_pre_activ):
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input
        if is_2d:
            batch_size = deriv_pre_activ.size(0)
            dLdZ = deriv_pre_activ * batch_size

            pe_grad_weight = torch.bmm(dLdZ.view(batch_size, -1, 1),
                                       H.view(batch_size, 1, -1))
            pe_grad_bias = dLdZ
        else:
            dLdZ = deriv_pre_activ.permute(1, 2, 0)
            dLdZ *= dLdZ.size(0)    
            pe_grad_weight = torch.bmm(dLdZ,
                                       H.transpose(0, 1))
            pe_grad_bias = dLdZ.sum(dim=-1)

        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        """
        Parameters:
        -------------------
        deriv_pre_activ: derivative of cost function w.r.t. the pre-activation of layer
        """
        is_2d = self.layer_input.dim() == 2
        H = self.layer_input

        if is_2d:
            batch_size = deriv_pre_activ.size(0)
            dLdZ = deriv_pre_activ * batch_size

            zsum = dLdZ.pow(2).sum(1)
            hsum = H.pow(2).sum(1)
            s = zsum * hsum
            
            return s + zsum
        else:
            pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)
            batch_size = pe_grad_weight.size(0)
            sq_norm_weight = pe_grad_weight.pow(2).view(batch_size, -1).sum(1)
            sq_norm_bias = pe_grad_bias.pow(2).view(batch_size, -1).sum(1)

            return sq_norm_weight + sq_norm_bias
