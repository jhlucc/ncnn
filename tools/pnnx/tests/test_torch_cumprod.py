# Copyright 2021 Tencent
# Copyright 2023 Xiaomi Corp.   (author: Fangjun Kuang)
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        # x - 3d
        # y - 2d
        # z - 1d
        x0 = torch.cumprod(x, dim=0)
        x1 = torch.cumprod(x, dim=1)
        x2 = torch.cumprod(x, dim=2)

        y0 = torch.cumprod(y, dim=0)
        y1 = torch.cumprod(y, dim=1)

        z0 = torch.cumprod(z, dim=0)
        return x0, x1, x2, y0, y1, z0

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(2, 3, 16)
    y = torch.rand(5, 9)
    z = torch.rand(14)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_cumprod.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_torch_cumprod.pt inputshape=[2,3,16],[5,9],[14]")

    # pnnx inference
    import test_torch_cumprod_pnnx
    b = test_torch_cumprod_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
