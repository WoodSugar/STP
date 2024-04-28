# -*- coding: utf-8 -*-
"""
@Time   : 2023/9/12

@Author : Shen Fang
"""
# Summary: Test if Pytorch Env supports CUDA and cudnn

import torch

if __name__ == '__main__':
    
    x = torch.Tensor([1.0])
    if torch.cuda.is_available():
        print("CUDA version: ", torch.version.cuda)
        print("CUDA device count: ", torch.cuda.device_count())
        print("CUDA device name: ", torch.cuda.get_device_name(0))
        print("CUDA device: ", torch.cuda.get_device_properties(0))
        print("CUDA device capability: ", torch.cuda.get_device_capability(0))
        print("CUDA device compute capability: ", torch.cuda.get_device_capability(0))
        xx = x.cuda()
    else:
        xx = x
    print("xx: ", xx)

    y = torch.randn(2, 3)
    if torch.cuda.is_available():
        yy = y.cuda()
    else:
        yy = y
    print("yy: ", yy)

    zz = xx + yy
    print("zz: ", zz)

    # CUDNN TEST
    from torch.backends import cudnn
    print("Support CUDA ?: ", torch.cuda.is_available())
    print("Support cudnn ?: ", cudnn.is_acceptable(xx))
