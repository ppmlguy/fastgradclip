import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_dataset(args, kwargs):
    if args.dname == 'cifar10':
        # loading data
        train_loader = DataLoader(
            datasets.CIFAR10(root=args.data_dir, train=True, download=args.download,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                      (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = DataLoader(
            datasets.CIFAR10(root=args.data_dir, train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                      (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dname == 'mnist':
        train_loader = DataLoader(
            datasets.MNIST(args.data_dir, train=True, download=args.download,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        test_loader = DataLoader(
            datasets.MNIST(args.data_dir, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dname == 'fmnist':
        train_loader = DataLoader(
            datasets.FashionMNIST(root=args.data_dir, train=True, download=args.download,
                                  transform = transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.1307,),
                                                           std=(0.3081,))
                                  ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = DataLoader(
            datasets.FashionMNIST(root=args.data_dir, train=False,
                                  transform = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = (0.1325,), std = (0.3105,))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dname == 'lsun':
        trans = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_loader = DataLoader(
            datasets.ImageFolder(root=os.path.join(args.data_dir, 'LSUN/train'), transform=trans),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = DataLoader(            
            datasets.ImageFolder(root=os.path.join(args.data_dir, 'LSUN/test'), transform=trans),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError('Unknown dataset')

    return train_loader, test_loader
