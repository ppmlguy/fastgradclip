import numpy as np
import torch
import torch.nn as nn


def verify_gradients(loss, model, pe_grads):
    full_grad = torch.autograd.grad(loss,
                                    filter(lambda x: x.requires_grad,
                                           model.parameters()),
                                    retain_graph=True)
    for fgrad in full_grad:
        print("grad.shape=", fgrad.shape)

    # verifying the gradients
    for i in range(len(pe_grads)):
        print('[{}] diff='.format(i),
              torch.norm(pe_grads[i].mean(dim=0) - full_grad[i]))


def clip_grad_norm(model, device, data, target, criterion, max_norm):
    model_params = [param for param in model.parameters() if param.requires_grad]
    grad = [torch.zeros(param.size(), device=device, requires_grad=False)
            for param in model_params]

    batch_size = len(data)

    for i in range(batch_size):
        output_i = model(data[i].unsqueeze(0))
        target_i = target[i].unsqueeze(0)

        loss = criterion(output_i, target_i)
        loss.backward()

        # bounding the total norm
        torch.nn.utils.clip_grad_norm_(model_params, max_norm)
        for j, param in enumerate(model_params):
            grad[j].add_(param.grad)

        model.zero_grad()

    return grad


def grad_per_loss(loss, model, device, batch_size, clip_thresh):
    clipped_grads = []
    losses = list(torch.split(loss, 1, dim=0))
    params = [param for param in model.parameters() if param.requires_grad]
    n_params = len(params)
    grads = [torch.autograd.grad(losses[i], params, retain_graph=True)
             for i in range(batch_size)]
    
    grad_norm = torch.zeros(batch_size, device=device, requires_grad=False)
    
    pe_grads = []
    for i in range(n_params):
        pe_grad = torch.stack(list(zip(*grads))[i], dim=0)
        # pe_grad = pe_grads[i]
        grad_norm += pe_grad.pow(2).view(batch_size, -1).sum(1)
        pe_grads.append(pe_grad)

    grad_norm.sqrt_()
    sample_weight = torch.min(torch.ones_like(grad_norm),
                              clip_thresh / grad_norm.detach())
    for i, param in enumerate(params):
        cut = sample_weight[(...,) + (None,)*(pe_grads[i].ndim-1)]
        clipped_grad = pe_grads[i] * cut
        clipped_grads.append(clipped_grad.mean(0))

    return clipped_grads


def test(models, device, criterion, data_iter):
    if not isinstance(models, list):
        models = [models]

    n_models = len(models)

    for model in models:
        model.eval()

    loss = np.zeros(n_models)
    correct = np.zeros(n_models, dtype=np.int32)

    with torch.no_grad():
        for data, target in data_iter:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]

            for i, model in enumerate(models):
                output = model(data)
                loss[i] += criterion(output, target).sum().item()
                pred = output.argmax(dim=1, keepdim=True)
                correct[i] += pred.eq(target.view_as(pred)).sum().item()

    total = len(data_iter.dataset)
    loss /= total
    acc = 100.0 * (correct / total)

    return loss, acc
