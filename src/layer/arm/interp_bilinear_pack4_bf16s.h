// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void resize_bilinear_image_pack4_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w, (size_t)4 * 4u, 4);
    Mat rowsbuf1(w, (size_t)4 * 4u, 4);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S1p = S1 + sx;

                float32x2_t _a01 = vld1_f32(alphap);

                float32x4_t _S10 = bfloat2float(vld1_u16(S1p));
                float32x4_t _S11 = bfloat2float(vld1_u16(S1p + 4));
                float32x4_t _rows1 = vmulq_lane_f32(_S10, _a01, 0);
                _rows1 = vmlaq_lane_f32(_rows1, _S11, _a01, 1);
                vst1q_f32(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned short* S0 = src.row<const unsigned short>(sy);
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;

                float32x2_t _a01 = vld1_f32(alphap);

                float32x4_t _S00 = bfloat2float(vld1_u16(S0p));
                float32x4_t _S01 = bfloat2float(vld1_u16(S0p + 4));
                float32x4_t _S10 = bfloat2float(vld1_u16(S1p));
                float32x4_t _S11 = bfloat2float(vld1_u16(S1p + 4));
                float32x4_t _rows0 = vmulq_lane_f32(_S00, _a01, 0);
                float32x4_t _rows1 = vmulq_lane_f32(_S10, _a01, 0);
                _rows0 = vmlaq_lane_f32(_rows0, _S01, _a01, 1);
                _rows1 = vmlaq_lane_f32(_rows1, _S11, _a01, 1);
                vst1q_f32(rows0p + dx * 4, _rows0);
                vst1q_f32(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float32x2_t _b01 = vld1_f32(beta);

        float* rows0p = rows0;
        float* rows1p = rows1;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);
            float32x4_t _Dp = vmulq_lane_f32(_rows0, _b01, 0);
            _Dp = vmlaq_lane_f32(_Dp, _rows1, _b01, 1);
            vst1_u16(Dp, float2bfloat(_Dp));

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}
