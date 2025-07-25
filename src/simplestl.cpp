// Copyright 2018 Leo <leo@nullptr.com.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "platform.h"

#if NCNN_SIMPLESTL

#include <stdlib.h>

// allocation functions
void* operator new(size_t size)
{
    return malloc(size);
}

void* operator new[](size_t size)
{
    return malloc(size);
}

// placement allocation functions
void* operator new(size_t /*size*/, void* ptr)
{
    return ptr;
}

void* operator new[](size_t /*size*/, void* ptr)
{
    return ptr;
}

// deallocation functions
void operator delete(void* ptr)
{
    free(ptr);
}

void operator delete[](void* ptr)
{
    free(ptr);
}

// deallocation functions since c++14
#if __cplusplus >= 201402L

void operator delete(void* ptr, size_t sz)
{
    free(ptr);
}

void operator delete[](void* ptr, size_t sz)
{
    free(ptr);
}

#endif

// placement deallocation functions
void operator delete(void* /*ptr*/, void* /*voidptr2*/)
{
}

void operator delete[](void* /*ptr*/, void* /*voidptr2*/)
{
}

extern "C" void __cxa_pure_virtual()
{
    NCNN_LOGE("[Fatal] Pure virtual func called, now exit.");
    // do not abort here to avoid more unpredictable behaviour
}

#endif // NCNN_SIMPLESTL



// simplestl 是 ncnn 为了追求极致可移植性的终极武器。
//
// 解决了终极依赖: 它解决了对 C++ 运行时库 (libstdc++/libc++) 的依赖，使得 ncnn 有可能在非常底层的环境中运行。
// 实现最小化: 它只实现了 C++ 运行时最最基础、不可或缺的部分（内存管理和纯虚函数处理），没有引入任何复杂的功能。
// 体现了深度: 这部分代码表明 ncnn 的作者对 C++ 的底层工作原理（内存管理、虚函数表、ABI）有非常深刻的理解。
// 对于绝大多数在普通操作系统（Linux, Windows, macOS, Android, iOS）上使用 ncnn 的用户来说，NCNN_SIMPLESTL 宏通常是关闭的，ncnn 会正常使用系统提供的、功能更完整、性能可能也经过高度优化的 C++ 运行时库。
//
// simplestl 的存在，更多地是展示了 ncnn 的一种潜力和设计上的完备性——它已经为最极端的部署环境做好了准备。