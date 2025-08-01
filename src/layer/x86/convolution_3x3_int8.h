// Copyright 2019 BUG1989
// SPDX-License-Identifier: BSD-3-Clause

static void conv3x3s1_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    *outptr0 += sum0;

                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
        }
    }
}

static void conv3x3s2_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const signed char* kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* kernel0 = (const signed char*)kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

                    *outptr0 += sum0;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }
}
