// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"
#include <string.h>

namespace pnnx {

class nn_GRU_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
GRU                     gru         3 1 input W R out %*=%*
torch.squeeze           sqz         1 1 out out1 dim=%dim
pnnx.Output             output      1 0 out1
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.GRU";
    }

    const char* name_str() const
    {
        return "gru";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (captured_params.find("gru.hidden_size") == captured_params.end())
            return false;

        const int hidden_size = captured_params.at("gru.hidden_size").i;

        std::string direction = "forward";
        if (captured_params.find("gru.direction") != captured_params.end())
        {
            direction = captured_params.at("gru.direction").s;
        }

        if (direction != "forward" && direction != "bidirectional")
            return false;

        const int num_directions = direction == "bidirectional" ? 2 : 1;

        if (captured_params.find("gru.activations") != captured_params.end())
        {
            const std::vector<std::string>& acts = captured_params.at("gru.activations").as;

            if (num_directions == 1)
            {
                if (acts != std::vector<std::string>{"Sigmoid", "Tanh"})
                    return false;
            }
            else // if (num_directions == 2)
            {
                if (acts != std::vector<std::string>{"Sigmoid", "Tanh", "Sigmoid", "Tanh"})
                    return false;
            }
        }

        if (captured_params.find("dim") != captured_params.end())
        {
            if (captured_params.at("dim").type == 2 && captured_params.at("dim").i != 1)
                return false;

            if (captured_params.at("dim").type == 5 && captured_params.at("dim").ai != std::vector<int>{1})
                return false;
        }

        const auto& W = captured_attrs.at("W.data");
        const auto& R = captured_attrs.at("R.data");

        if (W.shape.size() != 3 || W.shape[0] != num_directions || W.shape[1] != 3 * hidden_size)
            return false;

        if (R.shape.size() != 3 || R.shape[0] != num_directions || R.shape[1] != 3 * hidden_size || R.shape[2] != hidden_size)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        std::string direction = "forward";
        if (captured_params.find("gru.direction") != captured_params.end())
        {
            direction = captured_params.at("gru.direction").s;
        }

        const auto& W = captured_attrs.at("W.data");
        const auto& R = captured_attrs.at("R.data");

        bool batch_first = false;
        if (captured_params.find("gru.layout") != captured_params.end())
        {
            const int layout = captured_params.at("gru.layout").i;
            batch_first = layout == 1;
        }

        const int hidden_size = captured_params.at("gru.hidden_size").i;

        const int input_size = W.shape[2];

        op->params["input_size"] = input_size;
        op->params["hidden_size"] = hidden_size;
        op->params["num_layers"] = 1;
        op->params["bias"] = false;
        op->params["batch_first"] = batch_first;
        op->params["bidirectional"] = direction == "bidirectional" ? true : false;

        // split W R and reorder URN to RUN
        auto W_data = W.get_float32_data();
        auto R_data = R.get_float32_data();

        std::vector<float> W2(3 * hidden_size * input_size);
        {
            const int weight_data_size_g = hidden_size * input_size;

            const float* uptr = (const float*)W_data.data();
            const float* rptr = (const float*)W_data.data() + weight_data_size_g;
            const float* nptr = (const float*)W_data.data() + weight_data_size_g * 2;

            float* w_rptr = (float*)W2.data();
            float* w_uptr = (float*)W2.data() + weight_data_size_g;
            float* w_nptr = (float*)W2.data() + weight_data_size_g * 2;

            memcpy(w_rptr, rptr, weight_data_size_g * sizeof(float));
            memcpy(w_uptr, uptr, weight_data_size_g * sizeof(float));
            memcpy(w_nptr, nptr, weight_data_size_g * sizeof(float));
        }

        std::vector<float> R2(3 * hidden_size * hidden_size);
        {
            const int weight_data_size_g = hidden_size * hidden_size;

            const float* uptr = (const float*)R_data.data();
            const float* rptr = (const float*)R_data.data() + weight_data_size_g;
            const float* nptr = (const float*)R_data.data() + weight_data_size_g * 2;

            float* w_rptr = (float*)R2.data();
            float* w_uptr = (float*)R2.data() + weight_data_size_g;
            float* w_nptr = (float*)R2.data() + weight_data_size_g * 2;

            memcpy(w_rptr, rptr, weight_data_size_g * sizeof(float));
            memcpy(w_uptr, uptr, weight_data_size_g * sizeof(float));
            memcpy(w_nptr, nptr, weight_data_size_g * sizeof(float));
        }

        if (direction == "bidirectional")
        {
            op->attrs["weight_ih_l0"] = Attribute({3 * hidden_size, input_size}, W2);
            op->attrs["weight_hh_l0"] = Attribute({3 * hidden_size, hidden_size}, R2);

            std::vector<float> W2R(3 * hidden_size * input_size);
            {
                const int weight_data_size_g = hidden_size * input_size;

                const float* uptr = (const float*)W_data.data() + weight_data_size_g * 3;
                const float* rptr = (const float*)W_data.data() + weight_data_size_g * 4;
                const float* nptr = (const float*)W_data.data() + weight_data_size_g * 5;

                float* w_rptr = (float*)W2R.data();
                float* w_uptr = (float*)W2R.data() + weight_data_size_g;
                float* w_nptr = (float*)W2R.data() + weight_data_size_g * 2;

                memcpy(w_rptr, rptr, weight_data_size_g * sizeof(float));
                memcpy(w_uptr, uptr, weight_data_size_g * sizeof(float));
                memcpy(w_nptr, nptr, weight_data_size_g * sizeof(float));
            }

            std::vector<float> R2R(3 * hidden_size * hidden_size);
            {
                const int weight_data_size_g = hidden_size * hidden_size;

                const float* uptr = (const float*)R_data.data() + weight_data_size_g * 3;
                const float* rptr = (const float*)R_data.data() + weight_data_size_g * 4;
                const float* nptr = (const float*)R_data.data() + weight_data_size_g * 5;

                float* w_rptr = (float*)R2R.data();
                float* w_uptr = (float*)R2R.data() + weight_data_size_g;
                float* w_nptr = (float*)R2R.data() + weight_data_size_g * 2;

                memcpy(w_rptr, rptr, weight_data_size_g * sizeof(float));
                memcpy(w_uptr, uptr, weight_data_size_g * sizeof(float));
                memcpy(w_nptr, nptr, weight_data_size_g * sizeof(float));
            }

            op->attrs["weight_ih_l0_reverse"] = Attribute({3 * hidden_size, input_size}, W2R);
            op->attrs["weight_hh_l0_reverse"] = Attribute({3 * hidden_size, hidden_size}, R2R);
        }
        else
        {
            op->attrs["weight_ih_l0"] = Attribute({3 * hidden_size, input_size}, W2);
            op->attrs["weight_hh_l0"] = Attribute({3 * hidden_size, hidden_size}, R2);
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx, 140)

class nn_GRU_onnx_B : public nn_GRU_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
GRU                     gru         4 1 input W R B out %*=%*
torch.squeeze           sqz         1 1 out out1 dim=%dim
pnnx.Output             output      1 0 out1
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (!nn_GRU_onnx::match(captured_params, captured_attrs))
            return false;

        const int hidden_size = captured_params.at("gru.hidden_size").i;

        std::string direction = "forward";
        if (captured_params.find("gru.direction") != captured_params.end())
        {
            direction = captured_params.at("gru.direction").s;
        }

        const int num_directions = direction == "bidirectional" ? 2 : 1;

        const auto& B = captured_attrs.at("B.data");

        if (B.shape.size() != 2 || B.shape[0] != num_directions || B.shape[1] != 6 * hidden_size)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        nn_GRU_onnx::write(op, captured_params, captured_attrs);

        const auto& B = captured_attrs.at("B.data");

        bool has_bias = false;
        for (auto b : B.get_float32_data())
        {
            if (b != 0.f)
            {
                has_bias = true;
                break;
            }
        }

        op->params["bias"] = has_bias;

        if (has_bias)
        {
            const int hidden_size = captured_params.at("gru.hidden_size").i;

            // split B and reorder URN to RUN
            auto B_data = B.get_float32_data();

            std::vector<float> B2(3 * hidden_size);
            std::vector<float> B3(3 * hidden_size);
            {
                const float* uptr = (const float*)B_data.data();
                const float* rptr = (const float*)B_data.data() + hidden_size;
                const float* nptr = (const float*)B_data.data() + hidden_size * 2;

                float* w_rptr = (float*)B2.data();
                float* w_uptr = (float*)B2.data() + hidden_size;
                float* w_nptr = (float*)B2.data() + hidden_size * 2;

                memcpy(w_rptr, rptr, hidden_size * sizeof(float));
                memcpy(w_uptr, uptr, hidden_size * sizeof(float));
                memcpy(w_nptr, nptr, hidden_size * sizeof(float));
            }
            {
                const float* uptr = (const float*)B_data.data() + hidden_size * 3;
                const float* rptr = (const float*)B_data.data() + hidden_size * 4;
                const float* nptr = (const float*)B_data.data() + hidden_size * 5;

                float* w_rptr = (float*)B3.data();
                float* w_uptr = (float*)B3.data() + hidden_size;
                float* w_nptr = (float*)B3.data() + hidden_size * 2;

                memcpy(w_rptr, rptr, hidden_size * sizeof(float));
                memcpy(w_uptr, uptr, hidden_size * sizeof(float));
                memcpy(w_nptr, nptr, hidden_size * sizeof(float));
            }

            std::string direction = "forward";
            if (captured_params.find("gru.direction") != captured_params.end())
            {
                direction = captured_params.at("gru.direction").s;
            }

            if (direction == "bidirectional")
            {
                op->attrs["bias_ih_l0"] = Attribute({3 * hidden_size}, B2);
                op->attrs["bias_hh_l0"] = Attribute({3 * hidden_size}, B3);

                std::vector<float> B2R(3 * hidden_size);
                std::vector<float> B3R(3 * hidden_size);
                {
                    const float* uptr = (const float*)B_data.data() + hidden_size * 6;
                    const float* rptr = (const float*)B_data.data() + hidden_size * 7;
                    const float* nptr = (const float*)B_data.data() + hidden_size * 8;

                    float* w_rptr = (float*)B2R.data();
                    float* w_uptr = (float*)B2R.data() + hidden_size;
                    float* w_nptr = (float*)B2R.data() + hidden_size * 2;

                    memcpy(w_rptr, rptr, hidden_size * sizeof(float));
                    memcpy(w_uptr, uptr, hidden_size * sizeof(float));
                    memcpy(w_nptr, nptr, hidden_size * sizeof(float));
                }
                {
                    const float* uptr = (const float*)B_data.data() + hidden_size * 9;
                    const float* rptr = (const float*)B_data.data() + hidden_size * 10;
                    const float* nptr = (const float*)B_data.data() + hidden_size * 11;

                    float* w_rptr = (float*)B3R.data();
                    float* w_uptr = (float*)B3R.data() + hidden_size;
                    float* w_nptr = (float*)B3R.data() + hidden_size * 2;

                    memcpy(w_rptr, rptr, hidden_size * sizeof(float));
                    memcpy(w_uptr, uptr, hidden_size * sizeof(float));
                    memcpy(w_nptr, nptr, hidden_size * sizeof(float));
                }

                op->attrs["bias_ih_l0_reverse"] = Attribute({3 * hidden_size}, B2R);
                op->attrs["bias_hh_l0_reverse"] = Attribute({3 * hidden_size}, B3R);
            }
            else
            {
                op->attrs["bias_ih_l0"] = Attribute({3 * hidden_size}, B2);
                op->attrs["bias_hh_l0"] = Attribute({3 * hidden_size}, B3);
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_B, 140)

class nn_GRU_onnx_1 : public nn_GRU_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
GRU                     gru         4 2 input W R initial_h out outh %*=%*
torch.squeeze           sqz         1 1 out out1 dim=%dim
pnnx.Output             output      2 0 out1 outh
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_1, 140)

class nn_GRU_onnx_B1 : public nn_GRU_onnx_B
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 9
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
GRU                     gru         5 2 input W R B initial_h out outh %*=%*
torch.squeeze           sqz         1 1 out out1 dim=%dim
pnnx.Output             output      2 0 out1 outh
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_B1, 140)

class nn_GRU_onnx_2 : public nn_GRU_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
GRU                     gru         4 1 input W R initial_h out %*=%*
torch.squeeze           sqz         1 1 out out1 dim=%dim
pnnx.Output             output      1 0 out1
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_2, 140)

class nn_GRU_onnx_B2 : public nn_GRU_onnx_B
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
GRU                     gru         5 1 input W R B initial_h out %*=%*
torch.squeeze           sqz         1 1 out out1 dim=%dim
pnnx.Output             output      1 0 out1
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_B2, 140)

class nn_GRU_onnx_3 : public nn_GRU_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
GRU                     gru         3 1 input W R out %*=%*
Tensor.permute          transpose   1 1 out out1 dims=(0,2,1,3)
Tensor.reshape          reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      1 0 out2
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (!nn_GRU_onnx::match(captured_params, captured_attrs))
            return false;

        if (captured_params.at("reshape.shape").ai != std::vector<int>{0, 0, -1})
            return false;

        return true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_3, 140)

class nn_GRU_onnx_B3 : public nn_GRU_onnx_B
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
GRU                     gru         4 1 input W R B out %*=%*
Tensor.permute          transpose   1 1 out out1 dims=(0,2,1,3)
Tensor.reshape          reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      1 0 out2
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (!nn_GRU_onnx_B::match(captured_params, captured_attrs))
            return false;

        if (captured_params.at("reshape.shape").ai != std::vector<int>{0, 0, -1})
            return false;

        return true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_B3, 140)

class nn_GRU_onnx_4 : public nn_GRU_onnx_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 9
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
GRU                     gru         4 2 input W R initial_h out outh %*=%*
Tensor.permute          transpose   1 1 out out1 dims=(0,2,1,3)
Tensor.reshape          reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      2 0 out2 outh
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_4, 140)

class nn_GRU_onnx_B4 : public nn_GRU_onnx_B3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 10
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
GRU                     gru         5 2 input W R B initial_h out outh %*=%*
Tensor.permute          transpose   1 1 out out1 dims=(0,2,1,3)
Tensor.reshape          reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      2 0 out2 outh
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_B4, 140)

class nn_GRU_onnx_5 : public nn_GRU_onnx_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
GRU                     gru         4 1 input W R initial_h out %*=%*
Tensor.permute          transpose   1 1 out out1 dims=(0,2,1,3)
Tensor.reshape          reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      1 0 out2
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_5, 140)

class nn_GRU_onnx_B5 : public nn_GRU_onnx_B3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
GRU                     gru         5 1 input W R B initial_h out %*=%*
Tensor.permute          transpose   1 1 out out1 dims=(0,2,1,3)
Tensor.reshape          reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      1 0 out2
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_GRU_onnx_B5, 140)

} // namespace pnnx
