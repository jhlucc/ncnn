// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RELU_H
#define LAYER_RELU_H

#include "layer.h"

namespace ncnn {

class ReLU : public Layer
{
public:
    ReLU();
//声明了 load_param ReLU 层现在重写了 load_param 方法。这暗示着它可以从 .param 文件中读取一些配置参数了。
    virtual int load_param(const ParamDict& pd);

    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

public:
    float slope; //斜率
};

} // namespace ncnn

#endif // LAYER_RELU_H
