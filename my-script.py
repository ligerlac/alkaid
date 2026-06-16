import torch
import torch.nn as nn
from torchlogix.layers import GroupSum, LogicConv2d, LogicConv3d, LogicDense, OrPooling2d
from torchlogix.circuit import Circuit
from torchlogix.utils import set_export_mode
from datetime import datetime
from alkaid.trace import FVArray, FVArrayInput, trace
from alkaid.trace.ops import einsum, quantize, relu
from alkaid.codegen import HLSModel, RTLModel
from alkaid.converter import trace_model
import numpy as np
# from torch.fx import symbolic_trace, GraphModule
from torch.fx.experimental.proxy_tensor import make_fx


class DenseModel(nn.Sequential):
    def __init__(self):
        self.input_shape = (32*32*3,)
        super().__init__(
            LogicDense(32*32*3, 2000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            LogicDense(2000, 2000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            GroupSum(10)
        )


# w/ custom forward pass. previously not possible to export
class BranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = LogicConv2d(in_dim=32, channels=3, num_kernels=8,
                    receptive_field_size=3, tree_depth=2,
                    parametrization_kwargs={"weight_init": "random"}) # 8 x 30 x 30 = 7200
        self.pool = OrPooling2d(kernel_size=2, stride=2) # 8 x 15 x 15 = 1800
        self.dense = LogicDense(1801, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"})
        self.group_sum = GroupSum(10)
        self.input_shape = (32*32*3 + 1,)

    def forward(self, x):
        assert x.shape[1:] == (32*32*3 + 1,)
        img, feat = x[:, :-1].reshape(-1, 3, 32, 32), x[:, -1:]
        x = self.conv(img)
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.cat([x, feat], dim=1)
        x = self.dense(x)
        x = self.group_sum(x)
        return x
    

# from the paper "Convolutional Differentiable Logic Gate Networks"
class ClgnCifar(nn.Sequential):
    k = None
    n_bits = None
    tau = None
    llkw = {"parametrization_kwargs": {"weight_init": "random"}}
    def __init__(self):
        self.input_shape = (3*self.n_bits, 32, 32)
        super().__init__(
            LogicConv2d(
                in_dim=32,
                num_kernels=self.k,
                channels=3*self.n_bits,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **self.llkw
            ),
            OrPooling2d(kernel_size=2, stride=2), # kx16x
            LogicConv2d(
                in_dim=16,
                channels=self.k,
                num_kernels=4*self.k,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **self.llkw
            ),
            OrPooling2d(kernel_size=2, stride=2),
            LogicConv2d(
                in_dim=8,
                channels=4*self.k,
                num_kernels=16*self.k,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **self.llkw
            ),
            OrPooling2d(kernel_size=2, stride=2),
            LogicConv2d(
                in_dim=4,
                channels=16*self.k,
                num_kernels=32*self.k,
                tree_depth=3,
                receptive_field_size=3,
                padding=1,
                **self.llkw
            ),
            OrPooling2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            LogicDense(in_dim=128*self.k, out_dim=1280*self.k, **self.llkw),
            LogicDense(in_dim=1280*self.k, out_dim=640*self.k, **self.llkw),
            LogicDense(in_dim=640*self.k, out_dim=320*self.k, **self.llkw),
            GroupSum(k=10, tau=self.tau),
        )


class ClgnCifarSmall(ClgnCifar):
    k = 16
    n_bits = 2
    tau = 20


class ClgnCifarMedium(ClgnCifar):
    k = 256
    n_bits = 2
    tau = 20


def _make_input(shapes, kif):
    k, i, f = kif
    out = []
    for s in shapes:
        shp = (1,) + tuple(s)
        out.append(
            FVArray.from_kif(
                np.full(shp, k, dtype=np.int8),
                np.full(shp, i, dtype=np.int8),
                np.full(shp, f, dtype=np.int8),
            )
        )
    return out[0] if len(out) == 1 else tuple(out)


if __name__ == "__main__":

    model = DenseModel()
    set_export_mode(model)

    # inp = FVArrayInput((1, *model.input_shape)).quantize(k=0, i=1, f=0)
    # out = model(inp)
    # comb = trace(inp, out)  # fails because of torch.empty_like (should be np.empty_like)


    # x_dummy = torch.zeros(1, *model.input_shape, dtype=torch.bool)
    # gm = make_fx(model)(x_dummy)
    # print(f"FX graph:\n{gm.graph}")

    inputs = FVArrayInput((1, *model.input_shape)).quantize(k=0, i=1, f=0)
    trace_inp, trace_out = trace_model(model, inputs=inputs, framework='torch')
    comb = trace(trace_inp, trace_out)
    print('CombLogic:', comb)

