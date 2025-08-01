# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z):
        x = torch.amin(x, dim=0, keepdim=False)
        y = torch.amin(y, dim=(1,2), keepdim=False)
        z = torch.amin(z, dim=(0,3), keepdim=True)
        return x, y, z

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(3, 16)
    y = torch.rand(5, 9, 11)
    z = torch.rand(8, 5, 9, 10)

    a = net(x, y, z)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z))
    mod.save("test_torch_amin.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_torch_amin.pt inputshape=[3,16],[5,9,11],[8,5,9,10]")

    # ncnn inference
    import test_torch_amin_ncnn
    b = test_torch_amin_ncnn.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
