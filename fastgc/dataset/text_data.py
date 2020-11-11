import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


def tokenize(s):
    return s.split(' ')


def load_imdb(args):    
    TEXT = data.Field(lower=True, tokenize=tokenize, batch_first=True,
                      fix_length=args.max_seq_len)
    LABEL = data.LabelField(dtype=torch.long)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=args.data_dir)

    # build a vocabulary
    TEXT.build_vocab(train_data, max_size=args.max_vocab_size-2,
                     vectors=GloVe(name='6B', dim=args.embedding_size))
    LABEL.build_vocab(train_data)

    train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data), batch_size=args.batch_size,
        sort_key=lambda x: len(x.text))

    n_token = len(TEXT.vocab)
    n_classes = len(LABEL.vocab)
    print("{} unique tokens in TEXT vocabulary".format(n_token))
    print("{} class labels".format(n_classes))

    return train_iter, test_iter, n_token, n_classes, TEXT.vocab.vectors
