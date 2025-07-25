// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_SIMPLEOMP_H
#define NCNN_SIMPLEOMP_H

#include "platform.h"

#if NCNN_SIMPLEOMP

#include <stdint.h>

// This minimal openmp runtime implementation only supports the llvm openmp abi
// and only supports #pragma omp parallel for num_threads(X)

#ifdef __cplusplus
extern "C" {
#endif

NCNN_EXPORT int omp_get_max_threads();

NCNN_EXPORT void omp_set_num_threads(int num_threads);

NCNN_EXPORT int omp_get_dynamic();

NCNN_EXPORT void omp_set_dynamic(int dynamic);

NCNN_EXPORT int omp_get_num_threads();

NCNN_EXPORT int omp_get_thread_num();

NCNN_EXPORT int kmp_get_blocktime();

NCNN_EXPORT void kmp_set_blocktime(int blocktime);

#ifdef __cplusplus
}
#endif

#endif // NCNN_SIMPLEOMP

#endif // NCNN_SIMPLEOMP_H



//  ncnn 工程化能力的又一个极致体现。
//
//解决了核心痛点: 它让 ncnn 的 CPU 并行计算能力不再受限于目标平台是否有 OpenMP 运行时库，极大地增强了 ncnn 的可移植性和鲁棒性。
//实现简洁高效: 它没有实现 OpenMP 的所有复杂功能（比如 reduction, task 等），只实现了 ncnn 中实际用到的最核心的并行 for 循环 (parallel for) 功能，保持了代码的轻量。
//技术深度: 它深入到了编译器 ABI 层面，通过实现 __kmpc_* 和 GOMP_* 这些底层接口，成功地“欺骗”了编译器，让编译器以为它在和真正的 OpenMP 库对话。
//通过阅读 simpleomp，你不仅了解了 ncnn 如何实现 CPU 并行，更重要的是，你看到了一个大型 C++ 项目是如何通过自研核心组件来解决复杂的跨平台和依赖性问题的。这是一种非常高级的工程思维。