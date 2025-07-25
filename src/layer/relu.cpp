// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu.h"

namespace ncnn {

ReLU::ReLU()
{
    one_blob_only = true;
    support_inplace = true;
}
//load_param 的实现
int ReLU::load_param(const ParamDict& pd)
{
    slope = pd.get(0, 0.f); //pd 是从 .param 文件解析来的参数字典。 请从字典里查找 key 为 0 的参数。如果找到了，就把它的值赋给 slope；如果没找到，就把默认值 0.0f 赋给 slope。

    return 0;
}

int ReLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    if (slope == 0.f) //relu
    {
        #pragma omp parallel for num_threads(opt.num_threads) // 这是一个 OpenMP 的编译器指令 (pragma)。OpenMP 是一个用于共享内存并行编程的 API。
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }

    return 0;
}

} // namespace ncnn
