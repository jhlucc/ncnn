// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "slice_arm.h"

#include "cpu.h"

namespace ncnn {

Slice_arm::Slice_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Slice_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = bottom_blobs[0].elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    const int* slices_ptr = slices;
    const int* indices_ptr = indices;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // slice vector
        int w = bottom_blob.w * elempack;
        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = 1;
#if __ARM_NEON
            if (opt.use_packing_layout)
            {
                out_elempack = slice % 4 == 0 ? 4 : 1;
            }
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const float* ptr = (const float*)bottom_blob + q;
            float* outptr = top_blob;
            memcpy(outptr, ptr, top_blob.w * top_blob.elemsize);

            q += slice;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // slice image height
        int w = bottom_blob.w;
        int h = bottom_blob.h * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((h - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = 1;
#if __ARM_NEON
            if (opt.use_packing_layout)
            {
                out_elempack = slice % 4 == 0 ? 4 : 1;
            }
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack, opt);
            if (bottom_blob_unpacked.empty())
                return -100;
        }

        const float* ptr = bottom_blob_unpacked;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

            if (out_elempack == 1 && top_blob.elempack == 4)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const float* r0 = ptr;
                    const float* r1 = ptr + w;
                    const float* r2 = ptr + w * 2;
                    const float* r3 = ptr + w * 3;

                    float* outptr0 = top_blob.row(j);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;

                        outptr0 += 4;
                    }

                    ptr += w * 4;
                }
            }
            else // if (out_elempack == 1 && top_blob.elempack == 1) if (out_elempack == 4 && top_blob.elempack == 4)
            {
                int size = w * top_blob.h;

                float* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);

                ptr += size * top_blob.elempack;
            }
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // slice image width
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            const float* ptr = bottom_blob.row(j);
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                float* outptr = top_blob.row(j);
                memcpy(outptr, ptr, top_blob.w * elemsize);

                ptr += top_blob.w * elempack;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // slice dim channel
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = channels - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? channels + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((channels - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = 1;
#if __ARM_NEON
            if (opt.use_packing_layout)
            {
                out_elempack = slice % 4 == 0 ? 4 : 1;
            }
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, d, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack, opt);
            if (bottom_blob_unpacked.empty())
                return -100;
        }

        int p = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

            if (out_elempack == 1 && top_blob.elempack == 4)
            {
                int size = top_blob.w * top_blob.h * top_blob.d;

                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* r0 = bottom_blob_unpacked.channel(p);
                    const float* r1 = bottom_blob_unpacked.channel(p + 1);
                    const float* r2 = bottom_blob_unpacked.channel(p + 2);
                    const float* r3 = bottom_blob_unpacked.channel(p + 3);

                    float* outptr0 = top_blob.channel(q);

                    for (int j = 0; j < size; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;

                        outptr0 += 4;
                    }

                    p += 4;
                }
            }
            else // if (out_elempack == 1 && top_blob.elempack == 1) if (out_elempack == 4 && top_blob.elempack == 4)
            {
                int size = top_blob.total();

                const float* ptr = bottom_blob_unpacked.channel(p);
                float* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);

                p += top_blob.c;
            }
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // slice dim height
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((h - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice, d, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const float* ptr = bottom_blob.channel(p);

            for (int j = 0; j < d; j++)
            {
                for (size_t i = 0; i < top_blobs.size(); i++)
                {
                    Mat& top_blob = top_blobs[i];

                    int size = top_blob.w * top_blob.h;

                    float* outptr = top_blob.channel(p).depth(j);
                    memcpy(outptr, ptr, size * elemsize);

                    ptr += size * elempack;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // slice dim width
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, d, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const float* ptr = bottom_blob.channel(p);

            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < h; k++)
                {
                    for (size_t i = 0; i < top_blobs.size(); i++)
                    {
                        Mat& top_blob = top_blobs[i];

                        float* outptr = top_blob.channel(p).depth(j).row(k);
                        memcpy(outptr, ptr, top_blob.w * elemsize);

                        ptr += top_blob.w * elempack;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = d - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? d + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((d - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, slice, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const float* ptr = bottom_blob.channel(p);

            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                int size = top_blob.w * top_blob.h * top_blob.d;

                float* outptr = top_blob.channel(p);
                memcpy(outptr, ptr, size * elemsize);

                ptr += size * elempack;
            }
        }
    }

    return 0;
}

int Slice_arm::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    const int* slices_ptr = slices;
    const int* indices_ptr = indices;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // slice vector
        int w = bottom_blob.w * elempack;
        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = 1;
#if __ARM_NEON
            if (opt.use_packing_layout)
            {
#if NCNN_ARM82
                out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && slice % 8 == 0 ? 8 : slice % 4 == 0 ? 4 : 1;
#else
                out_elempack = slice % 4 == 0 ? 4 : 1;
#endif
            }
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const unsigned short* ptr = (const unsigned short*)bottom_blob + q;
            unsigned short* outptr = top_blob;
            memcpy(outptr, ptr, top_blob.w * top_blob.elemsize);

            q += slice;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // slice image height
        int w = bottom_blob.w;
        int h = bottom_blob.h * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((h - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = 1;
#if __ARM_NEON
            if (opt.use_packing_layout)
            {
#if NCNN_ARM82
                out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && slice % 8 == 0 ? 8 : slice % 4 == 0 ? 4 : 1;
#else
                out_elempack = slice % 4 == 0 ? 4 : 1;
#endif
            }
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack, opt);
            if (bottom_blob_unpacked.empty())
                return -100;
        }

        const unsigned short* ptr = bottom_blob_unpacked;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

#if NCNN_ARM82
            if (out_elempack == 4 && top_blob.elempack == 8)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const unsigned short* r0 = ptr;
                    const unsigned short* r1 = ptr + w * 4;

                    unsigned short* outptr0 = top_blob.row<unsigned short>(j);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = r0[0];
                        outptr0[1] = r0[1];
                        outptr0[2] = r0[2];
                        outptr0[3] = r0[3];
                        outptr0[4] = r1[0];
                        outptr0[5] = r1[1];
                        outptr0[6] = r1[2];
                        outptr0[7] = r1[3];

                        r0 += 4;
                        r1 += 4;
                        outptr0 += 8;
                    }

                    ptr += w * 8;
                }
            }
            if (out_elempack == 1 && top_blob.elempack == 8)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const unsigned short* r0 = ptr;
                    const unsigned short* r1 = ptr + w;
                    const unsigned short* r2 = ptr + w * 2;
                    const unsigned short* r3 = ptr + w * 3;
                    const unsigned short* r4 = ptr + w * 4;
                    const unsigned short* r5 = ptr + w * 5;
                    const unsigned short* r6 = ptr + w * 6;
                    const unsigned short* r7 = ptr + w * 7;

                    unsigned short* outptr0 = top_blob.row<unsigned short>(j);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;
                        outptr0[4] = *r4++;
                        outptr0[5] = *r5++;
                        outptr0[6] = *r6++;
                        outptr0[7] = *r7++;

                        outptr0 += 8;
                    }

                    ptr += w * 8;
                }
            }
#endif // NCNN_ARM82
            if (out_elempack == 1 && top_blob.elempack == 4)
            {
                for (int j = 0; j < top_blob.h; j++)
                {
                    const unsigned short* r0 = ptr;
                    const unsigned short* r1 = ptr + w;
                    const unsigned short* r2 = ptr + w * 2;
                    const unsigned short* r3 = ptr + w * 3;

                    unsigned short* outptr0 = top_blob.row<unsigned short>(j);

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;

                        outptr0 += 4;
                    }

                    ptr += w * 4;
                }
            }
            if (out_elempack == top_blob.elempack) // 1-1 4-4 8-8
            {
                int size = w * top_blob.h;

                unsigned short* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);

                ptr += size * top_blob.elempack;
            }
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // slice image width
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            const unsigned short* ptr = bottom_blob.row<const unsigned short>(j);
            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                unsigned short* outptr = top_blob.row<unsigned short>(j);
                memcpy(outptr, ptr, top_blob.w * elemsize);

                ptr += top_blob.w * elempack;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // slice dim channel
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c * elempack;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = channels - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? channels + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((channels - q) / (top_blobs.size() - i));
                }
            }

            int out_elempack = 1;
#if __ARM_NEON
            if (opt.use_packing_layout)
            {
#if NCNN_ARM82
                out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && slice % 8 == 0 ? 8 : slice % 4 == 0 ? 4 : 1;
#else
                out_elempack = slice % 4 == 0 ? 4 : 1;
#endif
            }
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, d, slice / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        size_t out_elemsize = top_blobs[0].elemsize;
        int out_elempack = top_blobs[0].elempack;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            out_elemsize = std::min(out_elemsize, top_blobs[i].elemsize);
            out_elempack = std::min(out_elempack, top_blobs[i].elempack);
        }

        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > out_elempack)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, out_elempack, opt);
            if (bottom_blob_unpacked.empty())
                return -100;
        }

        int p = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            Mat& top_blob = top_blobs[i];

#if NCNN_ARM82
            if (out_elempack == 4 && top_blob.elempack == 8)
            {
                int size = top_blob.w * top_blob.h * top_blob.d;

                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob_unpacked.channel(p);
                    const unsigned short* r1 = bottom_blob_unpacked.channel(p + 1);

                    unsigned short* outptr0 = top_blob.channel(q);

                    for (int j = 0; j < size; j++)
                    {
                        outptr0[0] = r0[0];
                        outptr0[1] = r0[1];
                        outptr0[2] = r0[2];
                        outptr0[3] = r0[3];
                        outptr0[4] = r1[0];
                        outptr0[5] = r1[1];
                        outptr0[6] = r1[2];
                        outptr0[7] = r1[3];

                        r0 += 4;
                        r1 += 4;
                        outptr0 += 8;
                    }

                    p += 2;
                }
            }
            if (out_elempack == 1 && top_blob.elempack == 8)
            {
                int size = top_blob.w * top_blob.h * top_blob.d;

                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob_unpacked.channel(p);
                    const unsigned short* r1 = bottom_blob_unpacked.channel(p + 1);
                    const unsigned short* r2 = bottom_blob_unpacked.channel(p + 2);
                    const unsigned short* r3 = bottom_blob_unpacked.channel(p + 3);
                    const unsigned short* r4 = bottom_blob_unpacked.channel(p + 4);
                    const unsigned short* r5 = bottom_blob_unpacked.channel(p + 5);
                    const unsigned short* r6 = bottom_blob_unpacked.channel(p + 6);
                    const unsigned short* r7 = bottom_blob_unpacked.channel(p + 7);

                    unsigned short* outptr0 = top_blob.channel(q);

                    for (int j = 0; j < size; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;
                        outptr0[4] = *r4++;
                        outptr0[5] = *r5++;
                        outptr0[6] = *r6++;
                        outptr0[7] = *r7++;

                        outptr0 += 8;
                    }

                    p += 8;
                }
            }
#endif // NCNN_ARM82
            if (out_elempack == 1 && top_blob.elempack == 4)
            {
                int size = top_blob.w * top_blob.h * top_blob.d;

                for (int q = 0; q < top_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob_unpacked.channel(p);
                    const unsigned short* r1 = bottom_blob_unpacked.channel(p + 1);
                    const unsigned short* r2 = bottom_blob_unpacked.channel(p + 2);
                    const unsigned short* r3 = bottom_blob_unpacked.channel(p + 3);

                    unsigned short* outptr0 = top_blob.channel(q);

                    for (int j = 0; j < size; j++)
                    {
                        outptr0[0] = *r0++;
                        outptr0[1] = *r1++;
                        outptr0[2] = *r2++;
                        outptr0[3] = *r3++;

                        outptr0 += 4;
                    }

                    p += 4;
                }
            }
            if (out_elempack == top_blob.elempack) // 1-1 4-4 8-8
            {
                int size = top_blob.total();

                const unsigned short* ptr = bottom_blob_unpacked.channel(p);
                unsigned short* outptr = top_blob;
                memcpy(outptr, ptr, size * top_blob.elemsize);

                p += top_blob.c;
            }
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // slice dim height
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = h - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? h + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((h - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, slice, d, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const unsigned short* ptr = bottom_blob.channel(p);

            for (int j = 0; j < d; j++)
            {
                for (size_t i = 0; i < top_blobs.size(); i++)
                {
                    Mat& top_blob = top_blobs[i];

                    int size = top_blob.w * top_blob.h;

                    unsigned short* outptr = top_blob.channel(p).depth(j);
                    memcpy(outptr, ptr, size * elemsize);

                    ptr += size * elempack;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // slice dim width
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = w - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? w + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((w - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(slice, h, d, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const unsigned short* ptr = bottom_blob.channel(p);

            for (int j = 0; j < d; j++)
            {
                for (int k = 0; k < h; k++)
                {
                    for (size_t i = 0; i < top_blobs.size(); i++)
                    {
                        Mat& top_blob = top_blobs[i];

                        unsigned short* outptr = top_blob.channel(p).depth(j).row<unsigned short>(k);
                        memcpy(outptr, ptr, top_blob.w * elemsize);

                        ptr += top_blob.w * elempack;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;

        int q = 0;
        for (size_t i = 0; i < top_blobs.size(); i++)
        {
            int slice;
            if (indices_ptr)
            {
                if (i == top_blobs.size() - 1)
                {
                    slice = d - q;
                }
                else
                {
                    int indice = indices_ptr[i];
                    int positive_indice = indice < 0 ? d + indice : indice;
                    slice = positive_indice - q;
                }
            }
            else
            {
                slice = slices_ptr[i];
                if (slice == -233)
                {
                    slice = static_cast<int>((d - q) / (top_blobs.size() - i));
                }
            }

            Mat& top_blob = top_blobs[i];
            top_blob.create(w, h, slice, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            q += slice;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < channels; p++)
        {
            const unsigned short* ptr = bottom_blob.channel(p);

            for (size_t i = 0; i < top_blobs.size(); i++)
            {
                Mat& top_blob = top_blobs[i];

                int size = top_blob.w * top_blob.h * top_blob.d;

                unsigned short* outptr = top_blob.channel(p);
                memcpy(outptr, ptr, size * elemsize);

                ptr += size * elempack;
            }
        }
    }

    return 0;
}

} // namespace ncnn
