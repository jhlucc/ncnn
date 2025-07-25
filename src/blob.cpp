// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "blob.h"

namespace ncnn {

Blob::Blob()
{
    producer = -1;
    consumer = -1;
}

} // namespace ncnn


// 好的，我们来看一下 `blob.h` 和 `blob.cpp`。这是 ncnn 中一个非常简单，但又至关重要的数据结构。
//
// 在 ncnn 的语境中，一个 **Blob** (Binary Large Object) 指的就是神经网络中，层与层之间流动的**中间数据**。在代码实现上，ncnn 用 `ncnn::Mat` 来存储这些数据，而用 `ncnn::Blob` 来存储这些数据的**元信息 (Metadata)**。
//
// ### 1. `Blob` 的作用是什么？
//
// 我们之前在分析 `Net::load_param` 时已经接触到了 `Blob`。`Net` 类内部有一个 `std::vector<Blob> blobs` 列表。这个列表和 `std::vector<Layer*> layers` 列表一起，构成了整个网络的**静态计算图 (Static Computational Graph)**。
//
// 一个 `Blob` 对象，它**不存储**真正的像素或特征值数据（数据存在 `Extractor` 的 `blob_mats` 里），它只存储描述这个数据的**“身份信息”**。
//
// **一句话总结：`Blob` 是 ncnn 计算图中连接 `Layer` 的“节点”，它记录了某个中间 `Mat` 的名字、生产者、消费者和形状信息。**
//
// ---
//
// ### 2. `blob.h` - Blob 的定义
//
// 我们来看一下 `Blob` 类的成员变量，就能明白它的全部作用了。
//
// ```cpp
// // in blob.h
// class NCNN_EXPORT Blob
// {
// public:
//     Blob();
//
// public:
// #if NCNN_STRING
//     // 1. blob 的名字
//     std::string name;
// #endif // NCNN_STRING
//
//     // 2. 生产这个 blob 的层的索引
//     int producer;
//
//     // 3. 消费这个 blob 的层的索引
//     int consumer;
//
//     // 4. 形状提示
//     Mat shape;
// };
// ```
//
// **成员变量解读**:
//
// 1.  **`std::string name;`**:
//     *   这个 Blob 的唯一标识符，比如 "conv1", "pool10", "prob"。
//     *   这个名字直接来自于 `.param` 文件。
//     *   它是 `Net` 在构建图时查找和连接节点，以及用户在 `Extractor` 中通过 `input()` 和 `extract()` 指定输入输出的依据。
//
// 2.  **`int producer;`**:
//     *   记录了**哪个 `Layer` 生成了这个 `Blob`**。
//     *   它的值是这个 `Layer` 在 `Net::d->layers` 向量中的**索引**。
//     *   例如，对于名为 "conv1" 的 `Blob`，它的 `producer` 就是 `Convolution` 层 `conv1` 的索引。
//     *   这个信息在 `Net::forward_layer` 的递归计算中至关重要。当需要计算某个 `Blob` 时，ncnn 就是通过它的 `producer` 索引找到对应的 `Layer` 去执行 `forward`。
//
// 3.  **`int consumer;`**:
//     *   记录了**哪个 `Layer` 将这个 `Blob` 作为输入**。
//     *   它的值是这个 `Layer` 在 `Net::d->layers` 向量中的索引。
//     *   **注意**: 原始的 ncnn 设计中，一个 `Blob` 只能有一个 `consumer`。如果一个 `Blob` 需要被多个后续层使用，ncnn 会通过一个 `Split` 层将它复制成多份，每一份都有一个单独的 `consumer`。这就是为什么你在 `squeezenet.param` 中会看到 `Split` 层。
//     *   这个 `consumer` 信息主要用于 `lightmode` 下的内存优化。当 `consumer` 层执行完毕后，ncnn 知道这个 `Blob` 已经没有用处了，就可以安全地回收它的内存。
//
// 4.  **`Mat shape;`**:
//     *   这是一个**形状提示 (shape hint)**。
//     *   这个 `Mat` 对象是空的，它不包含任何数据（`data` 指针为 `NULL`）。它只用来存储这个 `Blob` 的维度信息（`w, h, c, dims` 等）。
//     *   这个信息有什么用？在某些需要预先知道输入形状的层中，`Layer` 可以在 `load_param` 或 `create_pipeline` 阶段，通过 `bottom_shapes`（一个 `Mat` 的向量）提前获取到输入 `Blob` 的形状，从而进行一些预计算或准备工作。
//
// ---
//
// ### 3. `blob.cpp` - 简单的构造函数
//
// ```cpp
// // in blob.cpp
// #include "blob.h"
//
// namespace ncnn {
//
// Blob::Blob()
// {
//     producer = -1;
//     consumer = -1;
// }
//
// } // namespace ncnn
// ```
//
// *   **解读**: `Blob` 的实现非常简单。它的构造函数只是将 `producer` 和 `consumer` 初始化为 `-1`，表示这是一个新创建的、尚未连接到任何 `Layer` 的 `Blob`。
//
// ### 总结
//
// `Blob` 是一个看似简单但设计精巧的类，它在 ncnn 的图结构中扮演着“胶水”的角色。
//
// 1.  **图的节点**: `Layer` 是图的“边”（代表操作），`Blob` 则是图的“节点”（代表数据）。
// 2.  **元信息载体**: 它不存数据，只存**元信息**——名字、连接关系（生产者/消费者）、形状。
// 3.  **连接的桥梁**: 通过 `producer` 和 `consumer` 索引，`Blob` 将 `Layer` 之间解耦，并清晰地定义了数据流动的方向。
// 4.  **推理调度的依据**: `Net` 的推理调度器 `forward_layer` 正是依赖 `Blob` 的 `producer` 信息来找到需要执行的 `Layer`。
// 5.  **内存优化的基础**: `consumer` 信息是 `lightmode` 内存优化的关键。
//
// 现在，你已经彻底理解了 ncnn 的静态计算图是如何由 `std::vector<Layer*>` 和 `std::vector<Blob>` 这两个核心列表构建起来的。当你再次看到 `Net::load_param` 时，你脑海中应该能浮现出一张由 `Layer` 和 `Blob` 连接而成的、清晰的网络拓扑图。