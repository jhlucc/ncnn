// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_convolutiondepthwise_activation.h"

#include "pass_level2.h"

#include <float.h>

namespace pnnx {

namespace ncnn {

class fuse_convolutiondepthwise_relu_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
ConvolutionDepthWise    op_0        1 1 input a %*=%*
ReLU                    op_1        1 1 a out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ConvolutionDepthWise";
    }

    const char* name_str() const
    {
        return "convdwrelu";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        float slope = 0.f;
        if (captured_params.find("op_1.0") != captured_params.end())
        {
            slope = captured_params.at("op_1.0").f;
        }

        if (slope == 0.f)
        {
            op->params["9"] = 1;
        }
        else
        {
            op->params["9"] = 2;
            op->params["10"] = Parameter{slope};
        }
    }
};

class fuse_convolutiondepthwise_clip_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
ConvolutionDepthWise    op_0        1 1 input a %*=%*
Clip                    op_1        1 1 a out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ConvolutionDepthWise";
    }

    const char* name_str() const
    {
        return "convdwclip";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        float min = -FLT_MAX;
        float max = FLT_MAX;
        if (captured_params.find("op_1.0") != captured_params.end())
        {
            min = captured_params.at("op_1.0").f;
        }
        if (captured_params.find("op_1.1") != captured_params.end())
        {
            max = captured_params.at("op_1.1").f;
        }

        op->params["9"] = 3;
        op->params["10"] = Parameter{min, max};
    }
};

class fuse_convolutiondepthwise_sigmoid_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
ConvolutionDepthWise    op_0        1 1 input a %*=%*
Sigmoid                 op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ConvolutionDepthWise";
    }

    const char* name_str() const
    {
        return "convdwsigmoid";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 4;
    }
};

class fuse_convolutiondepthwise_mish_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
ConvolutionDepthWise    op_0        1 1 input a %*=%*
Mish                    op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ConvolutionDepthWise";
    }

    const char* name_str() const
    {
        return "convdwmish";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.find("op_0.9") == captured_params.end();
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        for (const auto& p : captured_params)
        {
            const std::string& pkey = p.first;
            const Parameter& pp = p.second;

            if (pkey.substr(0, 5) == "op_0.")
                op->params[pkey.substr(5)] = pp;
        }

        for (const auto& a : captured_attrs)
        {
            const std::string& akey = a.first;
            const Attribute& ap = a.second;

            if (akey.substr(0, 5) == "op_0.")
                op->attrs[akey.substr(5)] = ap;
        }

        op->params["9"] = 5;
    }
};

void fuse_convolutiondepthwise_activation(Graph& graph)
{
    fuse_convolutiondepthwise_relu_pass a;
    fuse_convolutiondepthwise_clip_pass b;
    fuse_convolutiondepthwise_sigmoid_pass c;
    fuse_convolutiondepthwise_mish_pass d;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
}

} // namespace ncnn

} // namespace pnnx
