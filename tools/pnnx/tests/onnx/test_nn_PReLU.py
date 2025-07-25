# Copyright 2024 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.prelu_0 = nn.PReLU(num_parameters=12)
        self.prelu_1 = nn.PReLU(num_parameters=1, init=0.12)

    def forward(self, x, y, z, w):
        x = x * 2 - 1
        y = y * 2 - 1
        z = z * 2 - 1
        w = w * 2 - 1

        x = self.prelu_0(x)
        x = self.prelu_1(x)

        y = self.prelu_0(y)
        y = self.prelu_1(y)

        z = self.prelu_0(z)
        z = self.prelu_1(z)

        w = self.prelu_0(w)
        w = self.prelu_1(w)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12)
    y = torch.rand(1, 12, 64)
    z = torch.rand(1, 12, 24, 64)
    w = torch.rand(1, 12, 24, 32, 64)

    a = net(x, y, z, w)

    # export onnx
    torch.onnx.export(net, (x, y, z, w), "test_nn_PReLU.onnx")

    # onnx to pnnx
    import os
    os.system("../../src/pnnx test_nn_PReLU.onnx inputshape=[1,12],[1,12,64],[1,12,24,64],[1,12,24,32,64]")

    # pnnx inference
    import test_nn_PReLU_pnnx
    b = test_nn_PReLU_pnnx.test_inference()

    for a0, b0 in zip(a, b):
        if not torch.equal(a0, b0):
            return False
    return True

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
