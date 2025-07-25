// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "option.h"

#include "cpu.h"

namespace ncnn {

Option::Option()
{
    lightmode = true;
    use_shader_pack8 = false;
    use_subgroup_ops = false;
    use_reserved_0 = false;

    num_threads = get_physical_big_cpu_count();
    blob_allocator = 0;
    workspace_allocator = 0;

#if NCNN_VULKAN
    blob_vkallocator = 0;
    workspace_vkallocator = 0;
    staging_vkallocator = 0;
    pipeline_cache = 0;
#endif // NCNN_VULKAN

    openmp_blocktime = 20;

    use_winograd_convolution = true;
    use_sgemm_convolution = true;
    use_int8_inference = true;
    use_vulkan_compute = false; // TODO enable me

    use_bf16_storage = false;

    use_fp16_packed = true;
    use_fp16_storage = true;
    use_fp16_arithmetic = true;
    use_int8_packed = true;
    use_int8_storage = true;
    use_int8_arithmetic = false;

    use_packing_layout = true;

    vulkan_device_index = -1;
    use_reserved_1 = false;

    use_tensor_storage = false;
    use_reserved_1p = false;

    use_reserved_2 = false;

    flush_denormals = 3;

    use_local_pool_allocator = true;

    use_shader_local_memory = true;
    use_cooperative_matrix = true;

    use_winograd23_convolution = true;
    use_winograd43_convolution = true;
    use_winograd63_convolution = true;

    use_a53_a55_optimized_kernel = is_current_thread_running_on_a53_a55();

    use_fp16_uniform = true;
    use_int8_uniform = true;

    use_reserved_9 = false;
    use_reserved_10 = false;
    use_reserved_11 = false;
}

} // namespace ncnn

// 在 load_param 和 load_model 过程中，Net 会读取 opt 的配置，来决定如何创建 Layer（比如是创建 Layer_final 代理，还是纯 CPU Layer）。
// 在 create_pipeline 时，opt 会被传递给每个 Layer，Layer 根据 opt 来初始化自己的计算管线（比如 GPU Layer 会根据 use_fp16_... 来选择不同的 shader）。
// 在 forward 时，opt 会被再次传递，Layer 根据它来决定运行时的行为（比如使用多少个线程）。