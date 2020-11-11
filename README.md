# Fast Per-Example Gradient Clipping
This repository contains the source code for the paper "Scaling Up Differentially Private Deep Learning with Fast Per-Example Gradient Clipping" (to appear in POPETS 2021).

The provided `fastgc` package provides a fast and scalable PyTorch implementation of *gradient clipping* method used for satisfying differential privacy. It provides a set of wrapper classes for neural network layers. To compute per-example gradients for differential privacy, you can simply import the `fastgc` package and replace the original PyTorch layers in your network with the provided wrapper classes.
The support types of layers include: 
- Linear layer
- Convolutional layer 
- Reccurent layer
- LayerNorm layer
- Transformer encoding layer


----------------------------------------------------------

## Installing dependecies
- The code was written and tested using Python 3.7 and Pytorch 1.5.0.
- To install the required packages, run 
```shell
pip install -r requirements.txt
```
- The code expects that `fastgc` package is available in your system. This means that `fastgc` directory is in your python's search path. An easy way to meet this requirement is to place `fastgc` directory as a subdirectory of one of directories in your `PYTHONPATH` environment variable.


## Running the code
The main function is implemented in `run_algo.py`. To train a neural network using the proposed ReweightGP algorithm, you can execute
```shell
python run_algo.py --train_alg reweight --data_dir <path to the directory containing the dataset> --download True
```
You will need to make sure the `--data_dir` is correctly set to the location containing the datasets to use. If the dataset to use is not available on the machine, please set `---download True` so that the program can download it from the internet.
If one or multiple GPUs are available on the machine, the program tries to choose a cuda device using `nvidia-smi` command. On Windows OS, it expects the executable is located in `C:\Program Files\NVIDIA Corporation\NVSMI` directory. On a Linux machine, it assumes the command is available in the current directory. Either add the location of `nvidia-smi` binary to the system's `PATH` environment variable or manually set the ID of cuda device using `--gpu_id` flag.

### Example usage
- Running non-private algorithm to train a CNN model on MNIST dataset
```shell
python run_algo.py --train_alg batch --model_name CNN --dname mnist --download True
```
- Running the proposed ReweightGP algorithm to train an RNN on mnist dataset, using mini-batches of size 128
```shell
python run_algo.py --train_alg reweight --model_name RNN --dname mnist --batch_size 128 --download True
```

## Important Input Arguments
The script takes several input arguments. To see available command line input arguments and allowed values, simply run 
```shell
python run_algo.py --help
```
Here we introduce each optional argument:
- `--data_dir`: string, path to the root directory in which input datasets are stored.
- `--dname`: the name of input dataset, Allowed values are {mnist, cifar10, fmnist, lsum}.
- `--train_alg`: the name of algorith to use for training, It can be `batch`, `reweight`, or `naive`.
- `--model_name`: the name of neural network architecture to use, Available values include:
  - `MLP`: a multi-layer perceptron
  - `CNN`: a convolutional neural network
  - `RNN`: a recurrent neural network
  - `LSTM`: an LSTM network
  - `Transformer`: an encoder block of Transformer network
  - ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`
  - VGG: `vgg11`, `vgg13`, `vgg16`, `vgg19`
- `--hidden_size`: an integer specifying the number of units in RNN
- `--hidden_sizes`: a list of integers, specifying the number of units in each layer of MLP model. For example, [128, 256] will create an MLP with two hidden layers with 128 and 256 hidden units, respectively.
- `--kernel_sizes`: a list of integers. This parameter sets the sizes of kernel in each layer of CNN model.
- `--fc_sizes`: a list of integers. This parameter is used to specify the number of hidden units in linear layers that are added after convolutional layers in the CNN model.
- `--batch_size`: integer, the number of observations in a mini-batch
- `--clip_thresh`: float, clipping threshold
- `--test_batch_size`: integer, how many observations to load in a mini-batch for testing
- `--act_func`: string, the name of activation function. Available values include: {`sigmoid`, `tanh`, `relu`, `lrelu`}
- `--optimizer`: string, name of optimization algorithm to use. There are 3 optimizers available: `Adam`, `SGD`, and `RMSprop`.
- `--lr`: float, learning rate (or step size)
- `--sigma`: float, scale parameter of noise distribution
- `--delta`: float, privacy parameter in $(\epsilon, \delta)$-differential privacy
- `--rep`: integer, how many times to repeat the experiment
- `--deterministic`: If this flag is set, the program fixes the randomness of numpy and pytorch (i.e., sets the seed to a fixed value)
- `--epochs`: integer, the number of epochs to run
- `--vervose`: If this option is provided, the program prints out detailed messages.
- `--img_size`: integer, this flag is used to set the image size of LSUN dataset. For example, if `img_size` is set to 64, the program resizes the images to 64 x 64.
- `--embedding_size`: integer, the length of embedding vectors for IMDB dataset (only used for Transformer model)
- `--max_vocab_size`: integer, the maximum number of words in the vocabulary (only used for Transformer model)
- `--num_layers`: integer, number of layers in a transformer block (only used for Transformer model)
- `--num_heads`: integer, number of attention head to put in a transformer block
- `--max_seq_len`: integer, maximum length of sequence for IMDB dataset
- `--niter`: integer, number of iterations to run the algorithm. If this parameter is given, the program ignores `epochs` parameter.
- `--download`: boolean (True or False) If true, the program will try to download the dataset from the internet if it is not found in the location specified by `data_dir`
- `--gpu_id`: integer, the id of cuda device to use

## Result Files
After the program runs for the specified number of iterations, it stores the results (avg. loss, avg. per-epoch execution time, etc) into a .csv file. The resulting csv file has the following naming convetion:
```shell
[training algorithm name]_[model name]_[dataset_name]_B[batch_size]E[epochs]C[clipping_threshold]SIG[sigma].csv
```

## Datasets
The paper uses 5 different datasets for the evaluation of proposed algorithm: 
- Pytorch built-in datasets: MNIST, FMNIST, CIFAR10, IMDB [API document](https://pytorch.org/docs/stable/torchvision/datasets.html)
- LSUN dataset can be downloaded from [here](https://www.yf.io/p/lsun).
