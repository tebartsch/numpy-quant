"""Adapated from https://github.com/renmengye/np-conv2d"""
import numpy as np


def array_offset(x):
    if x.base is None:
        return 0

    base_start = x.base.__array_interface__['data'][0]
    start = x.__array_interface__['data'][0]
    return start - base_start


def calc_size(h, kh, pad, sh):
    return int(np.ceil((h - kh + pad + 1) / sh))


def extract_sliding_windows(x, ksize, pad, stride, floor_first=True):
    """
    Args:
        x: [B, H, W, C]
        k: [KH, KW]
        pad: [PH0, PW0, PH1, PH2]
        stride: [SH, SW]
    Returns:
        y: [N, H, W, KH, KW, C]
    """
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]
    kh = ksize[0]
    kw = ksize[1]
    sh = stride[0]
    sw = stride[1]

    h2 = int(np.ceil((h - kh + pad[0] + pad[2] + 1) / sh))
    w2 = int(np.ceil((w - kw + pad[1] + pad[3] + 1) / sw))

    ph0 = pad[0]
    pw0 = pad[1]
    ph1 = pad[2]
    pw1 = pad[3]

    if floor_first:
        pph = (ph0, ph1)
        ppw = (pw0, pw1)
    else:
        pph = (ph1, ph0)
        ppw = (pw1, pw0)
    x = np.pad(
        x, ((0, 0), pph, ppw, (0, 0)),
        mode='constant',
        constant_values=(0.0,))

    # The following code extracts window without copying the data:
    # y = np.zeros([n, h2, w2, kh, kw, c])
    # for ii in range(h2):
    #     for jj in range(w2):
    #         xx = ii * sh
    #         yy = jj * sw
    #         y[:, ii, jj, :, :, :] = x[:, xx:xx + kh, yy:yy + kw, :]
    x_sn, x_sh, x_sw, x_sc = x.strides
    y_strides = (x_sn, sh * x_sh, sw * x_sw, x_sh, x_sw, x_sc)
    y = np.ndarray((n, h2, w2, kh, kw, c),
                   dtype=x.dtype,
                   buffer=x.data,
                   offset=array_offset(x),
                   strides=y_strides)
    return y


def conv2d(x, w, pad, stride):
    """"
    Args:
        x: [B, H, W, C]
        w: [I, J, C, K]
        pad: [PH, PW]
        stride: [SH, SW]

    Returns:
        y: [B, H, W, KH, KW, C]
    """
    ksize = w.shape[:2]
    x = extract_sliding_windows(x, ksize, pad, stride)
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    xs = x.shape
    x = x.reshape([xs[0] * xs[1] * xs[2], -1])
    y = x.dot(w)
    y = y.reshape([xs[0], xs[1], xs[2], -1])
    return y


def erf(x):
    """Source: https://stackoverflow.com/a/457805"""
    # save the sign of x
    sign = np.sign(x)
    x = np.abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y  # erf(-x) = -erf(x)
