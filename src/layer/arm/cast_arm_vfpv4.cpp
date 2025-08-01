// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "cast_fp16.h"

void cast_fp32_to_fp16_neon_vfpv4(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    cast_fp32_to_fp16_neon(bottom_blob, top_blob, opt);
}

void cast_fp16_to_fp32_neon_vfpv4(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    cast_fp16_to_fp32_neon(bottom_blob, top_blob, opt);
}

} // namespace ncnn
