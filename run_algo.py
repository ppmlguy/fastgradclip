import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import psutil
from fastgc.train import clip_grad_norm
from fastgc.train import grad_per_loss
from fastgc.train import test
from fastgc.common.opt import SGD
from fastgc.common.opt import Adam
from fastgc.util import check_gpu_memory
from fastgc.util import cuda_setup
from fastgc.util import io_setup
from fastgc.util import float_to_string
from fastgc.util import compute_epsilon
from fastgc.util import argument_parser
from fastgc.util import format_result
from fastgc.util import arrays_to_dataframe
from fastgc.util import get_filename
from fastgc.expr import create_model


def get_mem_usage(device):
    gpu_id = int(str(device).split(':')[1].strip())
    gpu_mem = check_gpu_memory()

    return gpu_mem[gpu_id]
    

def train_and_eval(device, train_loader, test_loader, input_size, output_size,
                   embeddings=None):
    model = create_model(args, input_size, output_size, embeddings).to(device)
    optim = Adam(filter(lambda x: x.requires_grad, model.parameters()),
                 lr=args.lr, sigma=args.sigma)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    total = len(train_loader.dataset)
    clip_thresh = args.clip_thresh
    epoch_mode = args.niter < 0

    if epoch_mode:
        tr_loss = np.zeros(args.epochs)
    else:
        tr_loss = np.zeros(args.niter)
        niter = 0

    tr_acc = np.zeros_like(tr_loss)
    te_loss = np.zeros_like(tr_loss)
    te_acc = np.zeros_like(tr_loss)
    etime = np.zeros_like(tr_loss)
    gpu_mem_max = np.zeros_like(tr_loss)
    gpu_mem = np.zeros_like(tr_loss)
    vir_mem = np.zeros_like(tr_loss)
    smi_mem = np.zeros_like(tr_loss)

    if args.verbose:
        header = "{0:5s} {1:>23s}".format('Epoch', args.train_alg)
        header_len = len(header)
        print(header)
        print('-'*header_len)

    for epoch in range(args.epochs):
        model.train()

        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = len(data)

            start_time = time.time()
            output = model(data)
            loss = criterion(output, target)
            mean_loss = loss.mean()
            mean_loss_val = mean_loss.item()
            optim.zero_grad()

            if not epoch_mode:
                gpu_mem[niter] = torch.cuda.memory_allocated(device) / float(2**20)                
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / float(2**20)

            if model.train_alg == 'reweight':
                with torch.no_grad():
                    grad_norm = model.pe_grad_norm(mean_loss, batch_size, device)
                    sample_weight = torch.min(torch.ones_like(grad_norm),
                                              clip_thresh / grad_norm.detach()) / batch_size
                loss.backward(sample_weight)
            elif model.train_alg == 'naive':
                grad = clip_grad_norm(model, device, data, target, criterion, clip_thresh)
    
                for j, param in enumerate(filter(lambda x: x.requires_grad, model.parameters())):
                    param.grad = grad[j] / batch_size
            elif model.train_alg == 'batch':
                mean_loss.backward()
            else:
                raise Exception('Unknown training algorithm')

            optim.step()
            elapsed = time.time() - start_time

            if epoch_mode:
                etime[epoch] += elapsed
            else:
                etime[niter] = elapsed
                gpu_mem_max[niter] = torch.cuda.max_memory_allocated(device) / (2**20) # MB
                smi_mem[niter] = get_mem_usage(device)
                tr_loss[niter] = mean_loss_val               
                vir_mem[niter] = mem
                
                if args.verbose:
                    print("{0:5s} {1:9.5f} {2:8.2f} {3:8.2f} {4:8.2f} {5:8.5f}".format(
                        '[' + str(niter) + ']', tr_loss[niter], gpu_mem[niter],
                        gpu_mem_max[niter], vir_mem[niter], elapsed))

                niter += 1

                if(niter % args.niter) == 0:
                    df = pd.DataFrame({'etime': etime, 'gpu_mem': gpu_mem,
                                       'cpu_mem': vir_mem, 'gpu_mem_max': gpu_mem_max,
                                       'smi_mem': smi_mem})
                    return df

        if epoch_mode:
            tr_loss[epoch], tr_acc[epoch] = test(model, device, criterion, train_loader)
            te_loss[epoch], te_acc[epoch] = test(model, device, criterion, test_loader)
            gpu_mem[epoch] = torch.cuda.memory_allocated(device) / (2**20) # MB
            gpu_mem_max[epoch] = torch.cuda.max_memory_allocated(device) / (2**20) # MB
            smi_mem[epoch] = get_mem_usage(device)

            if args.verbose:
                print("{0:5s} {1:9.5f} {2:6.2f} {3:6.3f}".format(
                    '[' + str(epoch+1) + ']', tr_loss[epoch], tr_acc[epoch], etime[epoch]))
                                         
                print("{0:5s} {1:9.5f} {2:6.2f}".format(' ', te_loss[epoch], te_acc[epoch]))
                print('-' * header_len)

    df = arrays_to_dataframe(tr_loss, tr_acc, te_loss, te_acc, etime)

    return df


def main():
    if args.deterministic:
        SEED = 1234
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    device, kwargs = cuda_setup(args.deterministic, args.gpu_id)
    train_loader, test_loader, input_size, output_size, embeddings = io_setup(args, kwargs)

    perf_data = []

    for i in range(args.rep):
        df = train_and_eval(device, train_loader, test_loader, input_size, output_size,
                            embeddings=embeddings)            
        perf_data.append(df)

    perf_df = pd.concat(perf_data, axis=0)

    filename = "{}_{}IMG{}.csv".format(args.train_alg, get_filename(args),
                                       args.img_size)

    if os.path.exists(filename):
        perf_df.to_csv(filename, mode='a', header=False, index=False)
    else:
        perf_df.to_csv(filename, index=False)


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()

    if args.data_dir is None:
        print('The path to the directory containing dataset must be set.')
        exit(-1)

    if args.model_name[:3] == 'vgg' or args.model_name[:3] == 'res':
        args.dname = 'lsun'

    if args.niter > 0:
        args.rep = 1

    if args.verbose:
        print("Parameters")
        print("----------")

        for arg in vars(args):
            print(" - {0:22s}: {1}".format(arg, getattr(args, arg)))
        print('\n')
    
    main()
