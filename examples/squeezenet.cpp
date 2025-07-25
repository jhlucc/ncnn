// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "net.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>

// ncnn 的作者 nihui 实现了一个名为 simpleocv 的小工具。它在 ncnn/tools/simpleocv 目录下，
// 只实现了 cv::Mat 数据结构和 cv::imread、cv::imwrite 等几个最最基础的函数，功能刚好够这些示例程序使用。它没有复杂的图像处理算法，代码量小，且没有任何其他依赖。
static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet; //创建了一个 ncnn::Net 对象。 一个空的神经网络框架。 后续的模型结构和权重都将加载到这个 squeezenet 对象中。这是 ncnn 的核心类之一。
    //ncnn 的一大优势就是跨平台的 GPU 加速。如果你的设备和驱动支持 Vulkan，ncnn 会自动使用 GPU 进行计算，速度会快很多。如果不支持，ncnn 会优雅地回退到 CPU 执行，所以这行代码是安全的。
    squeezenet.opt.use_vulkan_compute = true; //设置网络的一个选项。opt 是 ncnn::Option 类型的成员变量，用于控制网络的行为。这里尝试启用 Vulkan GPU 加速。

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    if (squeezenet.load_param("squeezenet_v1.1.param"))
        exit(-1);
    if (squeezenet.load_model("squeezenet_v1.1.bin"))
        exit(-1);
    //它从 OpenCV 的 cv::Mat 创建了一个 ncnn 的 ncnn::Mat。 是 ncnn 内部统一的数据容器，类似 PyTorch 的 Tensor。所有的计算都是基于 ncnn::Mat 进行的。
//告诉 ncnn，原始像素的通道顺序是 B-G-R。ncnn 内部会处理好。
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);
    //从Net对象创建了一个Extractor(提取器)对象。  Net 对象本身比较重，包含了完整的模型信息。而 Extractor 是一个轻量级的对象，专门用于执行一次前向推理 (forward pass)。
    //你可以把它想象成一个“推理会话”或“执行器”。设计成这样是线程安全的，你可以从同一个 Net 创建多个 Extractor 在不同线程里并行推理。 类似多个类
    ncnn::Extractor ex = squeezenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
// SqueezeNet 的输出 out 是一个 1x1x1000 的 Mat（在ncnn里表示为 w=1000, h=1, c=1）。out.w 就是类别数 1000
    cls_scores.resize(out.w);
    // ncnn::Mat 格式的输出结果，逐个复制到 C++ 标准的 std::vector<float> 中。out[j] 是访问 ncnn::Mat 数据的便捷方式。
    // 为什么？ 这样后续处理就和 ncnn 本身解耦了
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}
//表示这个函数的作用域仅限于当前文件 这是引用传递。意味着函数不会复制整个 cls_scores 向量，而是直接使用 main 函数中那个向量的“别名”。这样做效率很高，避免了大量数据的拷贝。
static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index获取分数的总数，这里是 1000。
    int size = cls_scores.size();
    //float 用来存储分数，int 用来存储这个分数对应的原始索引（类别ID）。
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }
//     vec = [
//   {0.0012, 0},  // 第0个元素: {类别0的分数, 索引0}
//   {0.005,  1},  // 第1个元素: {类别1的分数, 索引1}
//   {0.89,   2},  // 第2个元素: {类别2的分数, 索引2}
//   ...
//   {0.02, 999}   // 第999个元素: {类别999的分数, 索引999}
// ]

//使用 STL 的 partial_sort 对前 topk 个元素进行排序 的比较默认是先比较 first 元素，如果 first 相同再比较 second。所以这里会按照分数（float）进行降序排序。
//vec 的前 topk (3) 个元素，会是整个原始 vec 中分数最高的 3 个元素，并且这 3 个元素自身是排好序的（分数从高到低）。
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    //// 1. 检查命令行参数 argc 表示参数几个 argv 表示参数内容
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }
    // 2. 获取图片路径
    const char* imagepath = argv[1];
    // 读取图片 数组接受 flags 1表示为彩图
    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath); // 打印错误信息
        return -1;
    }
    // 定义向量容器保存结果
    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores); //5. 调用核心推理函数

    print_topk(cls_scores, 3);

    return 0;
}
