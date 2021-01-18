import argparse
import sys
import os
import subprocess
import numpy as np
import pandas as pd
import torch
from fastgc.dataset.img_data import load_dataset
from fastgc.dataset.text_data import load_imdb
from fastgc.activation import act_func_list


def check_gpu_memory():
    """
    Computes the size of available GPU memory.
    """
    curr_dir = os.getcwd()
    is_win32 = (sys.platform == "win32")

    if is_win32:
        # On Windows, it assumes 'nvidia-smi.exe' is available at this location
        nvsmi_dir = r"C:\Program Files\NVIDIA Corporation\NVSMI"
        os.chdir(nvsmi_dir)
        
    result = subprocess.check_output(['nvidia-smi',
                                      '--query-gpu=memory.used',
                                      '--format=csv,nounits,noheader'],
                                     encoding='utf-8')

    gpu_memory = [int(x) for x in result.strip().split('\n')]

    if is_win32:
        os.chdir(curr_dir)
    
    return gpu_memory

def cuda_setup(is_deterministic, gpu_idx=-1):
    """
    Create a device instance for pytorch. It uses a GPU if it is available.
    When multiple GPUs are available, it chooses the one with the largest available 
    gpu memory.
    """
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    if use_cuda:
        if gpu_idx < 0:
            # memory_usage = check_gpu_memory()
            # gpu_idx = np.argmin(memory_usage)
            gpu_idx = 0

        # torch.cuda.set_device(gpu_idx)
        device = torch.device("cuda:{}".format(gpu_idx))

        if not is_deterministic:
            torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    return device, kwargs


def copy_weight_values(models):
    model_states = [models[i].state_dict() for i in range(1, len(models))]

    for name, param in models[0].state_dict().items():
        for model_state in model_states:
            if name in model_state:
                model_state[name].copy_(param)


def clip_tensor(data, clip_thresh):
    """
    Clips all elements in the input tensor along the first dimension
    """
    batch_size = data.shape[0]
    
    x = data.view(batch_size, -1)
    data_norm = torch.norm(x, dim=1).detach()
    to_clip = data_norm > clip_thresh
    x[to_clip] = x[to_clip] * clip_thresh
    x[to_clip] /= data_norm[to_clip].unsqueeze(1)


def argument_parser():
    parser = argparse.ArgumentParser(description='fast per-example grad')

    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--dname', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'fmnist', 'lsun'])
    parser.add_argument('--train_alg', type=str, default='batch',
                        choices=['batch', 'reweight', 'naive'])
    parser.add_argument('--model_name', type=str, default='MLP',
                        choices=['MLP', 'CNN', 'RNN', 'LSTM', 'Transformer', 'resnet18',
                                 'resnet34', 'resnet50', 'resnet101', 'vgg11', 'vgg13',
                                 'vgg16', 'vgg19'])
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[128, 256])
    parser.add_argument('--channel_sizes', nargs='+', type=int, default=[20, 50])
    parser.add_argument('--kernel_sizes', nargs='+', type=int, default=[5, 5])
    parser.add_argument('--fc_sizes', nargs='+', type=int, default=[128])
    parser.add_argument('--batch_size', type=int, default=128)  
    parser.add_argument('--clip_thresh', type=float, default=1.0)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--act_func', type=str, default='sigmoid', choices=act_func_list)
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sigma', type=float, default=0.005)
    parser.add_argument('--delta', type=float, default=1e-5)
    # parser.add_argument('--beta1', type=float, default=0.9)
    # parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--rep', type=int, default=10)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--img_size', type=int, default=256)  # only for LSUN dataset

    # for transformer network
    parser.add_argument("--embedding_size", type=int, default=200,
                        help="length of embedding vectors")
    parser.add_argument("--max_vocab_size", type=int, default=50_000,
                        help="Number of words in the vocabulary.")
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--max_seq_len", help="Max sequence length.", default=512, type=int)
    parser.add_argument("--niter", type=int, default=-1)
    # parser.add_argument("--download", action='store_true')
    parser.add_argument("--download", type=bool, default=True)
    parser.add_argument("--gpu_id", help="gpu_id to use", default=-1, type=int)

    return parser


def float_to_string(value):
    str_val = "{0}".format(value).replace('.', '')

    return str_val


def conv_outsize(in_size, kernel_size, padding, stride):
    """
    Computes the size of output image after the convolution defined by the input arguments
    """
    out_size = (in_size - kernel_size + (2 * padding)) // stride
    out_size += 1

    return out_size


def compute_epsilon(clip_thresh, delta, sigma, batch_size, epochs):
    """
    Computes the privacy parameters for differential privacy
    """
    if clip_thresh <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0

    sens = clip_thresh / batch_size    
    log_delta = np.log(1./delta)
    alpha = 1. + sigma*np.sqrt(2*log_delta) / (sens*epochs)
    rdp_eps = alpha * epochs*(sens**2) / (2*(sigma**2))
    eps =  rdp_eps + log_delta/(alpha-1.0)

    return alpha, rdp_eps, eps


def format_result(loss, acc, etime=None):
    if etime is None:
        str_result = ["{0:9.5f} {1:6.2f} {2:6s}".format(l, a, ' ')
                      for l, a in zip(loss, acc)]
    else:
        str_result = ["{0:9.5f} {1:6.2f} {2:6.3f}".format(l, a, t)
                      for l, a, t in zip(loss, acc, etime)]
    
    return '  '.join(str_result)


def io_setup(args, kwargs):
    # loading data
    if args.model_name == 'Transformer':
        train_loader, test_loader, input_size, output_size, embeddings = load_imdb(args)
        args.dname = 'IMDB'
    else:
        train_loader, test_loader = load_dataset(args, kwargs)
        embeddings = None

        if args.model_name == 'MLP':
            input_size = train_loader.dataset[0][0].view(-1).size(0)
        elif args.model_name in ('RNN', 'LSTM'):
            C, H, W = train_loader.dataset[0][0].shape
            input_size = C * W
        else:
            input_size = train_loader.dataset[0][0].size(1)  # assume C x H x W and H=W

        output_size = len(train_loader.dataset.classes)

        if args.model_name == 'CNN':
            args.channel_sizes = [train_loader.dataset[0][0].size(0)] + args.channel_sizes

    return train_loader, test_loader, input_size, output_size, embeddings


def arrays_to_dataframe(tr_loss, tr_acc, te_loss, te_acc, etime, eps=None):
    # converting into dataframes
    perf = {
        'tr_loss': tr_loss,
        'te_loss': te_loss,
        'tr_acc': tr_acc,
        'te_acc': te_acc,
        'etime': etime,
    }
    if eps is not None:
        perf['eps'] = eps

    df = pd.DataFrame(perf)
    
    df['epoch'] = range(1, len(df)+1)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
    return df


def get_filename(args):
    filename = '{}_{}_B{}E{}C{}SIG{}'.format(args.model_name,
                                             args.dname,
                                             args.batch_size,
                                             args.epochs,
                                             float_to_string(args.clip_thresh),
                                             float_to_string(args.sigma))

    return filename
