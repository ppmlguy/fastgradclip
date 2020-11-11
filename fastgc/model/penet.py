import torch
import torch.nn as nn


class PeGradNet(nn.Module):
    def __init__(self):
        super(PeGradNet, self).__init__()
        self.layers = None

    def per_example_gradient(self, loss):
        """
        Computes per-example gradients
        """
        grads = []

        pre_acts = [m.pre_activation for m in self.layers]
        Z_grad = torch.autograd.grad(loss, pre_acts, retain_graph=True)

        for layer, zgrad in zip(self.layers, Z_grad):
            grads.extend(layer.per_example_gradient(zgrad))

        return grads

    def pe_grad_norm(self, loss, batch_size, device, block_size=-1):
        # a container to store the norms of per-exmaple graidents
        grad_norm = torch.zeros(batch_size, device=device, requires_grad=False)
        
        pre_acts = [layer.pre_activation for layer in self.layers]
        # computes the gradient of cost function w.r.t. pre-activations
        Z_grad = torch.autograd.grad(loss, pre_acts, retain_graph=True)

        for layer, zgrad in zip(self.layers, Z_grad):
            grad_norm.add_(layer.pe_grad_sqnorm(zgrad))
        
        grad_norm.sqrt_()

        return grad_norm
