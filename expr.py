from fastgc.model import MLP
from fastgc.model import CNN
from fastgc.model import RNN
from fastgc.model import SimpleLSTM
from fastgc.model import TransformerModel
from fastgc.model import resnet18
from fastgc.model import resnet34
from fastgc.model import resnet50
from fastgc.model import resnet101
from fastgc.model import vgg11
from fastgc.model import vgg13
from fastgc.model import vgg16
from fastgc.model import vgg19


def create_model(args, input_size, output_size, embeddings=None, train_alg=None):
    if train_alg is None:
        train_alg = args.train_alg

    # models
    if args.model_name == 'MLP':
        model = MLP(input_size, args.hidden_sizes, output_size,
                    act_func=args.act_func, train_alg=train_alg)
        args.num_layers = len(args.hidden_sizes)
    elif args.model_name == 'CNN':
        model = CNN(input_size, args.channel_sizes, args.kernel_sizes,
                     args.fc_sizes, output_size, train_alg=train_alg)
        args.num_layers = len(args.channel_sizes) + len(args.fc_sizes)
    elif args.model_name == 'RNN':
        model = RNN(input_size, args.hidden_size, output_size, train_alg=train_alg)
        args.num_layers = 1
    elif args.model_name == 'LSTM':
        model = SimpleLSTM(input_size, args.hidden_size, output_size, train_alg=train_alg)
        args.num_layers = 1
    elif args.model_name == 'Transformer':
        # n_token = input_size
        d_model = args.embedding_size
        n_heads = args.num_heads
        n_hidden = d_model * 4
        
        model = TransformerModel(input_size, output_size, d_model,
                                 args.num_layers, args.num_heads, n_hidden=n_hidden,
                                 embeddings=embeddings, train_alg=train_alg)
    elif args.model_name == 'resnet18':
        model = resnet18(pretrained=True, progress=False, num_classes=output_size,
                         train_alg=args.train_alg)
        args.num_layers = 18
    elif args.model_name == 'resnet34':
        model = resnet34(pretrained=True, progress=False, num_classes=output_size,
                         train_alg=args.train_alg)
        args.num_layers = 34
    elif args.model_name == 'resnet50':
        model = resnet50(pretrained=True, progress=False, num_classes=output_size,
                         train_alg=args.train_alg)
        args.num_layers = 50
    elif args.model_name == 'resnet101':
        model = resnet101(pretrained=True, progress=False, num_classes=output_size,
                          train_alg=args.train_alg)
        args.num_layers = 101
    elif args.model_name == 'vgg11':
        model = vgg11(pre_trained=False, progress=False, num_classes=output_size,
                      train_alg=args.train_alg)
        args.num_layers = 11
    elif args.model_name == 'vgg13':
        model = vgg13(pre_trained=False, progress=False, num_classes=output_size,
                      train_alg=args.train_alg)
        args.num_layers = 13
    elif args.model_name == 'vgg16':
        model = vgg16(pre_trained=False, progress=False, num_classes=output_size,
                      train_alg=args.train_alg)
        args.num_layers = 16
    elif args.model_name == 'vgg19':
        model = vgg19(pre_trained=False, progress=False, num_classes=output_size,
                      train_alg=args.train_alg)
        args.num_layers = 19
        
    return model


def create_models(args, device, input_size, output_size, embeddings, algos):
    # models
    models = [create_model(args, input_size, output_size, embeddings,
                           train_alg=algo).to(device) for algo in algos]


    return models
    
