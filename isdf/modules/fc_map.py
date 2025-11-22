# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from .smcper.entropy_layers import EntropyLinear, Custom_Entropy_Linear
from .smcper.parameter_decoders import AffineDecoder


def gradient(inputs, outputs):
    # print(outputs)
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return points_grad


def chunks(
    pc,
    chunk_size,
    fc_sdf_map,
    to_cpu=False,
    use_entropy=False,
):
    n_pts = pc.shape[0]
    n_chunks = int(np.ceil(n_pts / chunk_size))
    alphas = []
    for n in range(n_chunks):
        start = n * chunk_size
        end = start + chunk_size
        chunk = pc[start:end, :]
        if use_entropy:
            alpha, _ = fc_sdf_map(chunk)
        else:
            alpha = fc_sdf_map(chunk)

        alpha = alpha.squeeze(dim=-1)
        if to_cpu:
            alpha = alpha.cpu()
        alphas.append(alpha)

    alphas = torch.cat(alphas, dim=-1)

    return alphas


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f), torch.nn.Softplus(beta=100)
    )


def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    if isinstance(m, torch.nn.Linear):
        init_fn(m.weight)


class SDFMap(nn.Module):
    def __init__(
        self,
        positional_encoding,
        hidden_size=256,
        hidden_layers_block=1,
        scale_output=1.0,
    ):
        super(SDFMap, self).__init__()
        self.scale_output = scale_output

        self.positional_encoding = positional_encoding
        embedding_size = self.positional_encoding.embedding_size

        self.in_layer = fc_block(embedding_size, hidden_size)

        hidden1 = [
            fc_block(hidden_size, hidden_size) for _ in range(hidden_layers_block)
        ]
        self.mid1 = torch.nn.Sequential(*hidden1)

        self.cat_layer = fc_block(hidden_size + embedding_size, hidden_size)

        hidden2 = [
            fc_block(hidden_size, hidden_size) for _ in range(hidden_layers_block)
        ]
        self.mid2 = torch.nn.Sequential(*hidden2)

        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        self.apply(init_weights)

    def forward(self, x, noise_std=None, pe_mask=None, sdf1=None):
        x_pe = self.positional_encoding(x)
        if pe_mask is not None:
            x_pe = torch.mul(x_pe, pe_mask)

        fc1 = self.in_layer(x_pe)
        fc2 = self.mid1(fc1)
        fc2_x = torch.cat((fc2, x_pe), dim=-1)
        fc3 = self.cat_layer(fc2_x)
        fc4 = self.mid2(fc3)
        raw = self.out_alpha(fc4)

        if noise_std is not None:
            noise = torch.randn(raw.shape, device=x.device) * noise_std
            raw = raw + noise
        alpha = raw * self.scale_output

        return alpha.squeeze(-1)


# def entropy_fc_block(in_f, out_f, wdec, bdec=None, ema_decay=0):
#     return torch.nn.Sequential(
#         EntropyLinear(in_f, out_f, wdec, bdec, ema_decay),
#         torch.nn.Softplus(beta=100)
#     )
class EntropySDFMap(nn.Module):
    def __init__(
        self,
        positional_encoding,
        hidden_size=256,
        hidden_layers_block=1,
        scale_output=1.0,
        encode_bias=False,
    ):
        super(EntropySDFMap, self).__init__()
        self.scale_output = scale_output

        self.positional_encoding = positional_encoding
        embedding_size = self.positional_encoding.embedding_size

        self.in_wdec = AffineDecoder(1)
        self.mid1_wdec = AffineDecoder(1)
        self.cat_wdec = AffineDecoder(1)
        self.mid2_wdec = AffineDecoder(1)
        self.out_wdec = AffineDecoder(1)

        self.bdec = None
        if encode_bias:
            self.bdec = AffineDecoder(1)

        # self.bdec = AffineDecoder(1)
        self.ema_decay = 0

        self.in_layer = EntropyLinear(
            embedding_size, hidden_size, self.in_wdec, self.bdec, self.ema_decay
        )
        self.activation = nn.Softplus(beta=100)

        # hidden1 = [EntropyLinear(hidden_size, hidden_size, self.mid1_wdec, self.bdec, self.ema_decay)
        #            for _ in range(hidden_layers_block)]
        # self.mid1 = torch.nn.Sequential(*hidden1)

        self.mid1 = EntropyLinear(
            hidden_size, hidden_size, self.mid1_wdec, self.bdec, self.ema_decay
        )

        self.cat_layer = EntropyLinear(
            hidden_size + embedding_size,
            hidden_size,
            self.cat_wdec,
            self.bdec,
            self.ema_decay,
        )

        # hidden2 = [EntropyLinear(hidden_size, hidden_size, self.mid2_wdec, self.bdec, self.ema_decay)
        #            for _ in range(hidden_layers_block)]
        # self.mid2 = torch.nn.Sequential(*hidden2)
        self.mid2 = EntropyLinear(
            hidden_size, hidden_size, self.mid2_wdec, self.bdec, self.ema_decay
        )

        self.out_alpha = EntropyLinear(
            hidden_size, 1, self.out_wdec, None, self.ema_decay
        )
        self.layers = [
            self.in_layer,
            self.mid1,
            self.cat_layer,
            self.mid2,
            self.out_alpha,
        ]
        self.decoders = [
            self.in_wdec,
            self.mid1_wdec,
            self.cat_wdec,
            self.mid2_wdec,
            self.out_wdec,
        ]
        if encode_bias:
            self.decoders.append(self.bdec)

    def get_non_entropy_params(self):
        # 获取所有层的非熵编码参数
        layer_params = [layer.get_non_entropy_parameters() for layer in self.layers]

        # 获取所有解码器的参数

        decoder_params = [
            param for decoder in self.decoders for param in decoder.parameters()
        ]

        # if self.bdec is not None:
        #     decoder_params += list(self.bdec.parameters())

        # 合并所有参数
        return sum(layer_params, []) + decoder_params

    def get_decompressed_params(self):
        # 获取所有层的非熵编码参数
        layer_params = [layer.get_decompressed_parameters() for layer in self.layers]
        # 展平嵌套的列表
        return sum(layer_params, [])

    def get_entropy_params(self):
        # 获取所有层的熵编码参数
        layer_params = [layer.get_entropy_parameters() for layer in self.layers]
        # 展平嵌套的列表
        return sum(layer_params, [])

    def compress(self):
        b = None
        for dec in self.decoders:
            # if first decoder
            if dec == self.in_wdec:
                b = dec.compress()
            else:
                b += dec.compress()

        for layer in self.layers:
            b += layer.compress()
        # print(b)
        return b

    def decompress(self, b):
        lb = len(b)
        for dec in self.decoders:
            b = dec.decompress(b)
        for layer in self.layers:
            b = layer.decompress(b)

    def get_rate(self):
        para_size = 0
        entropy_model_size = 0
        for layer in self.layers:
            p, t = layer.get_compressed_params_size()
            para_size += p
            entropy_model_size += t
        for dec in self.decoders:
            d = dec.get_model_size()
            entropy_model_size += d
        return para_size, entropy_model_size

    def get_original_size(self):
        para_size = 0
        for layer in self.layers:
            para_size += layer.get_model_size()
        return para_size

    def update(self, force=False, super_up=True,update_quantiles=False):
        for layer in self.layers:
            layer.update(force=force, super_up=super_up)

    def forward(self, x, noise_std=None, pe_mask=None, sdf1=None):
        x_pe = self.positional_encoding(x)
        if pe_mask is not None:
            x_pe = torch.mul(x_pe, pe_mask)
        fc1, rate1 = self.in_layer(x_pe)
        fc1 = self.activation(fc1)

        fc2, rate2 = self.mid1(fc1)
        fc2 = self.activation(fc2)
        fc2_x = torch.cat((fc2, x_pe), dim=-1)

        fc3, rate3 = self.cat_layer(fc2_x)
        fc3 = self.activation(fc3)

        fc4, rate4 = self.mid2(fc3)
        fc4 = self.activation(fc4)

        raw, rate5 = self.out_alpha(fc4)

        rate = rate1 + rate2 + rate3 + rate4 + rate5

        if noise_std is not None:
            noise = torch.randn(raw.shape, device=x.device) * noise_std
            raw = raw + noise
        alpha = raw * self.scale_output

        return alpha.squeeze(-1), rate


class Custom_EntropySDFMap(nn.Module):
    def __init__(
        self,
        positional_encoding,
        hidden_size=256,
        hidden_layers_block=1,
        scale_output=1.0,
    ):
        super(Custom_EntropySDFMap, self).__init__()
        self.scale_output = scale_output

        self.positional_encoding = positional_encoding
        embedding_size = self.positional_encoding.embedding_size

        self.in_layer = Custom_Entropy_Linear(embedding_size, hidden_size)
        self.activation = nn.Softplus(beta=100)

        # hidden1 = [Custom_Entropy_Linear(hidden_size, hidden_size)
        #            for _ in range(hidden_layers_block)]
        # self.mid1 = torch.nn.Sequential(*hidden1)

        self.mid1 = Custom_Entropy_Linear(hidden_size, hidden_size)

        self.cat_layer = Custom_Entropy_Linear(
            hidden_size + embedding_size, hidden_size
        )

        # hidden2 = [EntropyLinear(hidden_size, hidden_size, self.mid2_wdec, self.bdec, self.ema_decay)
        #            for _ in range(hidden_layers_block)]
        # self.mid2 = torch.nn.Sequential(*hidden2)
        self.mid2 = Custom_Entropy_Linear(hidden_size, hidden_size)

        self.out_alpha = Custom_Entropy_Linear(
            hidden_size,
            1,
        )
        self.layers = [
            self.in_layer,
            self.mid1,
            self.cat_layer,
            self.mid2,
            self.out_alpha,
        ]

    def get_non_entropy_params(self):
        # 获取所有层的非熵编码参数
        layer_params = [layer.get_non_entropy_parameters() for layer in self.layers]
        # 合并所有参数
        return sum(layer_params, [])

    def get_decompressed_params(self):
        # 获取所有层的非熵编码参数
        layer_params = [layer.get_decompressed_parameters() for layer in self.layers]
        # 展平嵌套的列表
        return sum(layer_params, [])

    def get_entropy_params(self):
        # 获取所有层的熵编码参数
        layer_params = [layer.get_entropy_parameters() for layer in self.layers]
        # 展平嵌套的列表
        return sum(layer_params, [])

    def compress(self):
        b = b""
        for layer in self.layers:
            b += layer.compress()
        # print(b)
        return b

    def decompress(self, b):
        lb = len(b)
        for layer in self.layers:
            b = layer.decompress(b)

    def get_rate(self):
        para_size = 0
        entropy_model_size = 0
        for layer in self.layers:
            p, t = layer.get_compressed_params_size()
            para_size += p
            entropy_model_size += t

        return para_size, entropy_model_size

    def get_original_size(self):
        para_size = 0
        for layer in self.layers:
            para_size += layer.get_model_size()
        return para_size

    def update(self, force=False, update_quantiles=False):
        for layer in self.layers:
            layer.update(force=force, update_quantiles=update_quantiles)


    def forward(self, x, noise_std=None, pe_mask=None, sdf1=None):
        x_pe = self.positional_encoding(x)
        if pe_mask is not None:
            x_pe = torch.mul(x_pe, pe_mask)
        fc1, rate1 = self.in_layer(x_pe)
        fc1 = self.activation(fc1)

        fc2, rate2 = self.mid1(fc1)
        fc2 = self.activation(fc2)
        fc2_x = torch.cat((fc2, x_pe), dim=-1)

        fc3, rate3 = self.cat_layer(fc2_x)
        fc3 = self.activation(fc3)

        fc4, rate4 = self.mid2(fc3)
        fc4 = self.activation(fc4)

        raw, rate5 = self.out_alpha(fc4)

        rate = rate1 + rate2 + rate3 + rate4 + rate5

        if noise_std is not None:
            noise = torch.randn(raw.shape, device=x.device) * noise_std
            raw = raw + noise
        alpha = raw * self.scale_output

        return alpha.squeeze(-1), rate


if __name__ == "__main__":
    from embedding import PostionalEncoding

    positional_encoding = PostionalEncoding(
        min_deg=0,
        max_deg=5,
        scale=0.05937489,
    )
    net = EntropySDFMap(positional_encoding)
    rand_input = torch.rand(10, 3)
    y, rate = net(rand_input)
    print(y, "\n", rate)
