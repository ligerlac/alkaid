import torch
import torch.nn as nn
from alkaid.trace import trace
from alkaid.converter import trace_model
from alkaid.trace import FVArrayInput
from torch.fx import Tracer
from torch.fx.experimental.proxy_tensor import make_fx


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.ndim == 2, x.shape
        return x[:, 0] & x[:, 1]
    

def my_np_function(x):
    assert x.ndim == 2, x.shape
    return x[:, 0] & x[:, 1]


if __name__ == "__main__":
    # alkaid's functional API works on the numpy function
    inp = FVArrayInput((1, 2)).quantize(k=0, i=1, f=0)
    out = my_np_function(inp)
    comb = trace(inp, out)  # works with numpy function

    # the same function cannot be traced by torch.fx.Tracer
    model = MyModel()
    try:
        graph = Tracer().trace(model)  # fails
    except Exception as e:
        print(f"torch.fx.Tracer failed as expected: {e}")

    # but non-strict tracing with make_fx works
    graph = make_fx(model)(torch.empty((1,2), dtype=torch.bool))
    graph.print_readable()  # works with torch model

