// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void deconvolution_packn_fp16s_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int maxk = kernel_w * kernel_h;

    const float* bias_data_ptr = bias_data;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        __fp16* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                vfloat32m2_t _sum = __riscv_vfmv_v_f_f32m2(0.f, vl);

                if (bias_data_ptr)
                {
                    _sum = __riscv_vle32_v_f32m2(bias_data_ptr + p * packn, vl);
                }

                const __fp16* kptr = (const __fp16*)weight_data_fp16.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys < 0 || sys % stride_h != 0)
                            continue;

                        int sy = sys / stride_h;
                        if (sy >= h)
                            continue;

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            const __fp16* sptr = m.row<const __fp16>(sy) + sx * packn;

                            int k = y * kernel_w + x;

                            for (int l = 0; l < packn; l++)
                            {
                                __fp16 val = *sptr++;
                                vfloat16m1_t _w0 = __riscv_vle16_v_f16m1(kptr + k * packn * packn + packn * l, vl);
                                _sum = __riscv_vfwmacc_vf_f32m2(_sum, val, _w0, vl);
                            }
                        }
                    }

                    kptr += maxk * packn * packn;
                }

                _sum = activation_ps(_sum, activation_type, activation_params, vl);

                __riscv_vse16_v_f16m1(outptr + j * packn, __riscv_vfncvt_f_f_w_f16m1(_sum, vl), vl);
            }

            outptr += outw * packn;
        }
    }
}

static void deconvolution_packn_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data_fp16, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int maxk = kernel_w * kernel_h;

    const __fp16* bias_data_ptr = bias_data_fp16;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        __fp16* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                vfloat16m1_t _sum = __riscv_vfmv_v_f_f16m1(0.f, vl);

                if (bias_data_ptr)
                {
                    _sum = __riscv_vle16_v_f16m1(bias_data_ptr + p * packn, vl);
                }

                const __fp16* kptr = (const __fp16*)weight_data_fp16.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys < 0 || sys % stride_h != 0)
                            continue;

                        int sy = sys / stride_h;
                        if (sy >= h)
                            continue;

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            const __fp16* sptr = m.row<const __fp16>(sy) + sx * packn;

                            int k = y * kernel_w + x;

                            for (int l = 0; l < packn; l++)
                            {
                                __fp16 val = *sptr++;
                                vfloat16m1_t _w0 = __riscv_vle16_v_f16m1(kptr + k * packn * packn + packn * l, vl);
                                _sum = __riscv_vfmacc_vf_f16m1(_sum, val, _w0, vl);
                            }
                        }
                    }

                    kptr += maxk * packn * packn;
                }

                _sum = activation_ps(_sum, activation_type, activation_params, vl);

                __riscv_vse16_v_f16m1(outptr + j * packn, _sum, vl);
            }

            outptr += outw * packn;
        }
    }
}
