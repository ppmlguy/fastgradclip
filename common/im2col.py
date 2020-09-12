import numpy as np
import torch
import torch.nn.functional as F


def get_im2col_indices(x_shape, filter_height, filter_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - filter_height) % stride == 0
    assert (W + 2 * padding - filter_height) % stride == 0
    out_height = int((H + 2 * padding - filter_height) / stride + 1)
    out_width = int((W + 2 * padding - filter_width) / stride + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, C)  # repeat i0 C times
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), filter_height * filter_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, filter_height, filter_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    # x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    x_padded = F.pad(x, (p, p, p, p), 'constant', 0)
    # x_padded = F.pad(x, (0, 0, 0, 0, p, p, p, p), 'constant', 0)

    k, i, j = get_im2col_indices(x.shape, filter_height, filter_width, padding, stride)
    # print("k={}, i={}, j={}".format(k, i, j))
    cols = x_padded[:, k, i, j]
    # N = x.shape[0]
    # C = x.shape[1]
    # cols = cols.view(N, filter_height * filter_width * C, -1)

    return cols


def col2im_indices(cols, x_shape, filter_height=3, filter_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, filter_height, filter_width, padding, stride)
    
    cols_reshaped = cols.reshape(C * filter_height * filter_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


if __name__ == "__main__":
    N = 2
    n_filter = 1
    filter_h = 2
    filter_w = 2
    h_in = 3
    w_in = 3
    x = np.arange(N * n_filter * h_in * w_in).reshape(N, n_filter, h_in, w_in)
    x = torch.from_numpy(x)
    col = im2col_indices(x, filter_h, filter_w, padding=0, stride=1)
    print(col.shape)

        
