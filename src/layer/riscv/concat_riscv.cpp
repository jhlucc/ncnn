// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "concat_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Concat_riscv::Concat_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
    support_fp16_storage = cpu_support_riscv_zfh();
#endif

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Concat_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = bottom_blobs[0].elembits();

#if NCNN_ZFH
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    int dims = bottom_blobs[0].dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w * bottom_blob.elempack;
        }

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = top_w % packn == 0 ? packn : 1;
        }
#endif
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            const float* ptr = bottom_blob;
            memcpy(outptr, ptr, bottom_blob.w * bottom_blob.elemsize);

            outptr += bottom_blob.w * bottom_blob.elempack;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_h += bottom_blob.h * bottom_blob.elempack;
        }

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = top_h % packn == 0 ? packn : 1;
        }
#endif
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, top_h / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        float* outptr = top_blob_unpacked;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

#if __riscv_vector
            if (bottom_blob.elempack == packn && elempack == 1)
            {
                const size_t vl = __riscv_vsetvl_e32m1(packn);

                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const float* r0 = bottom_blob.row(i);

                    float* outptr0 = outptr;

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(r0, vl);
                        __riscv_vsse32_v_f32m1(outptr0, w * sizeof(float), _p, vl);

                        r0 += packn;
                        outptr0 += 1;
                    }

                    outptr += w * packn;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == packn && elempack == packn)
#endif           // __riscv_vector
            {
                int size = w * bottom_blob.h;

                const float* ptr = bottom_blob;
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                outptr += size * bottom_blob.elempack;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* outptr = top_blob.row(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                const float* ptr = bottom_blob.row(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elempack;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;

        // total channels
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_channels += bottom_blob.c * bottom_blob.elempack;
        }

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = top_channels % packn == 0 ? packn : 1;
        }
#endif
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, top_channels / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, h, d, top_channels / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;

            top_blob_unpacked.dims = dims;
        }

        int p = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

#if __riscv_vector
            if (bottom_blob.elempack == packn && elempack == 1)
            {
                const size_t vl = __riscv_vsetvl_e32m1(packn);

                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const float* r0 = bottom_blob.channel(q);

                    float* outptr0 = top_blob_unpacked.channel(p);

                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(r0, vl);
                        __riscv_vsse32_v_f32m1(outptr0, top_blob_unpacked.cstep * sizeof(float), _p, vl);

                        r0 += packn;
                        outptr0 += 1;
                    }

                    p += packn;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == 4 && elempack == 4)
#endif           // __riscv_vector
            {
                int size = bottom_blob.total();

                const float* ptr = bottom_blob;
                float* outptr = top_blob_unpacked.channel(p);
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                p += bottom_blob.c;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    int size = bottom_blob.w * bottom_blob.h;

                    const float* ptr = bottom_blob.channel(q).depth(i);
                    memcpy(outptr, ptr, size * elemsize);

                    outptr += size * elempack;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (size_t b = 0; b < bottom_blobs.size(); b++)
                    {
                        const Mat& bottom_blob = bottom_blobs[b];

                        const float* ptr = bottom_blob.channel(q).depth(i).row(j);
                        memcpy(outptr, ptr, bottom_blob.w * elemsize);

                        outptr += bottom_blob.w * elempack;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        // interleave dim depth
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total depth
        int top_d = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_d += bottom_blob.d;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                const float* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elempack;
            }
        }
    }

    return 0;
}

int Concat_riscv::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#endif

    int dims = bottom_blobs[0].dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w * bottom_blob.elempack;
        }

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = top_w % packn == 0 ? packn : 1;
        }
#endif
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned short* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            const unsigned short* ptr = bottom_blob;
            memcpy(outptr, ptr, bottom_blob.w * bottom_blob.elemsize);

            outptr += bottom_blob.w * bottom_blob.elempack;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_h += bottom_blob.h * bottom_blob.elempack;
        }

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = top_h % packn == 0 ? packn : 1;
        }
#endif
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, top_h / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;
        }

        unsigned short* outptr = top_blob_unpacked;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

#if __riscv_vector
            if (bottom_blob.elempack == packn && elempack == 1)
            {
                const size_t vl = __riscv_vsetvl_e16m1(packn);

                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                    unsigned short* outptr0 = outptr;

                    for (int j = 0; j < w; j++)
                    {
                        vuint16m1_t _p = __riscv_vle16_v_u16m1(r0, vl);
                        __riscv_vsse16_v_u16m1(outptr0, w * sizeof(unsigned short), _p, vl);

                        r0 += packn;
                        outptr0 += 1;
                    }

                    outptr += w * packn;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == packn && elempack == packn)
#endif           // __riscv_vector
            {
                int size = w * bottom_blob.h;

                const unsigned short* ptr = bottom_blob;
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                outptr += size * bottom_blob.elempack;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                const unsigned short* ptr = bottom_blob.row<unsigned short>(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elempack;
            }
        }
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;

        // total channels
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            elemsize = std::min(elemsize, bottom_blob.elemsize);
            elempack = std::min(elempack, bottom_blob.elempack);
            top_channels += bottom_blob.c * bottom_blob.elempack;
        }

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = top_channels % packn == 0 ? packn : 1;
        }
#endif
        size_t out_elemsize = elemsize / elempack * out_elempack;

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, top_channels / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        Mat top_blob_unpacked = top_blob;
        if (elempack < out_elempack)
        {
            top_blob_unpacked.create(w, h, d, top_channels / elempack, elemsize, elempack, opt.workspace_allocator);
            if (top_blob_unpacked.empty())
                return -100;

            top_blob_unpacked.dims = dims;
        }

        int p = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

#if __riscv_vector
            if (bottom_blob.elempack == packn && elempack == 1)
            {
                const size_t vl = __riscv_vsetvl_e16m1(packn);

                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const unsigned short* r0 = bottom_blob.channel(q);

                    unsigned short* outptr0 = top_blob_unpacked.channel(p);

                    for (int i = 0; i < size; i++)
                    {
                        vuint16m1_t _p = __riscv_vle16_v_u16m1(r0, vl);
                        __riscv_vsse16_v_u16m1(outptr0, top_blob_unpacked.cstep * sizeof(unsigned short), _p, vl);

                        r0 += packn;
                        outptr0 += 1;
                    }

                    p += packn;
                }
            }
            else // if (bottom_blob.elempack == 1 && elempack == 1) if (bottom_blob.elempack == packn && elempack == packn)
#endif           // __riscv_vector
            {
                int size = bottom_blob.total();

                const unsigned short* ptr = bottom_blob;
                unsigned short* outptr = top_blob_unpacked.channel(p);
                memcpy(outptr, ptr, size * bottom_blob.elemsize);

                p += bottom_blob.c;
            }
        }

        // packing
        if (elempack < out_elempack)
        {
            convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
        }
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    int size = bottom_blob.w * bottom_blob.h;

                    const unsigned short* ptr = bottom_blob.channel(q).depth(i);
                    memcpy(outptr, ptr, size * elemsize);

                    outptr += size * elempack;
                }
            }
        }
    }

    if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int d = bottom_blobs[0].d;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.dims = dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (size_t b = 0; b < bottom_blobs.size(); b++)
                    {
                        const Mat& bottom_blob = bottom_blobs[b];

                        const unsigned short* ptr = bottom_blob.channel(q).depth(i).row<const unsigned short>(j);
                        memcpy(outptr, ptr, bottom_blob.w * elemsize);

                        outptr += bottom_blob.w * elempack;
                    }
                }
            }
        }
    }

    if (dims == 4 && positive_axis == 1)
    {
        // interleave dim depth
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;
        int elempack = bottom_blobs[0].elempack;

        // total depth
        int top_d = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_d += bottom_blob.d;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_d, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

                const unsigned short* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elempack;
            }
        }
    }

    return 0;
}

} // namespace ncnn
