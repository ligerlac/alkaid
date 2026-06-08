"""Tensor-method dispatch for fx ``call_method`` nodes.

Each handler is a tiny function decorated with ``@_method('name', ...)`` that
takes ``(receiver, *args, **kwargs)``. Mirrors the ``@_functional`` pattern in
``functional.py`` but keyed by string method name instead of by callable.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from alkaid.trace.ops import quantize

_method_map: dict[str, Callable] = {}


def _method(*names: str):
    def decorator(fn: Callable) -> Callable:
        for n in names:
            _method_map[n] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Shape / structural
# ---------------------------------------------------------------------------


@_method('view', 'reshape')
def _reshape(receiver: Any, *args: Any, **_kwargs: Any) -> Any:
    shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
    return receiver.reshape(shape)


@_method('permute')
def _permute(receiver: Any, *args: Any, **_kwargs: Any) -> Any:
    dims = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
    return np.transpose(receiver, tuple(dims))


@_method('transpose')
def _transpose(receiver: Any, dim0: int, dim1: int, **_kwargs: Any) -> Any:
    return np.swapaxes(receiver, dim0, dim1)


@_method('t')
def _t(receiver: Any, **_kwargs: Any) -> Any:
    return np.transpose(receiver)


@_method('flatten')
def _flatten(receiver: Any, *args: Any, **kwargs: Any) -> Any:
    start = args[0] if len(args) > 0 else kwargs.get('start_dim', 0)
    end = args[1] if len(args) > 1 else kwargs.get('end_dim', -1)
    end = end if end >= 0 else receiver.ndim + end
    new_shape = receiver.shape[:start] + (-1,) + receiver.shape[end + 1 :]
    return receiver.reshape(new_shape)


@_method('unsqueeze')
def _unsqueeze(receiver: Any, dim: int, **_kwargs: Any) -> Any:
    return np.expand_dims(receiver, axis=dim)


@_method('squeeze')
def _squeeze(receiver: Any, *args: Any, **_kwargs: Any) -> Any:
    if args:
        return np.squeeze(receiver, axis=args[0])
    return np.squeeze(receiver)


@_method('expand')
def _expand(receiver: Any, *args: Any, **_kwargs: Any) -> Any:
    out_shape = tuple(receiver.shape[i] if s == -1 else s for i, s in enumerate(args))
    return np.broadcast_to(receiver, out_shape)


@_method('expand_as')
def _expand_as(receiver: Any, other: Any, **_kwargs: Any) -> Any:
    return np.broadcast_to(receiver, tuple(other.shape))


@_method('repeat')
def _repeat(receiver: Any, *args: Any, **_kwargs: Any) -> Any:
    return np.tile(receiver, tuple(args))


@_method('size')
def _size(receiver: Any, *args: Any, **_kwargs: Any) -> Any:
    return receiver.shape if not args else receiver.shape[args[0]]


# ---------------------------------------------------------------------------
# No-ops in replay (dtype/device coercions, copies)
# ---------------------------------------------------------------------------


@_method('contiguous', 'clone', 'detach', 'type_as', 'float', 'double', 'int', 'long')
def _identity(receiver: Any, *_args: Any, **_kwargs: Any) -> Any:
    return receiver


@_method('to')
def _to(receiver: Any, *args: Any, **kwargs: Any) -> Any:
    import torch

    _torch_integers = {
        torch.uint8: (0, 8),
        torch.int8: (1, 7),
        torch.uint16: (0, 16),
        torch.int16: (1, 15),
        torch.uint32: (0, 32),
        torch.int32: (1, 31),
        torch.uint64: (0, 64),
        torch.int64: (1, 63),
    }
    dtype_to = kwargs.get('dtype', args[0] if args else None)
    if dtype_to == torch.bool:
        return receiver != 0
    if dtype_to in _torch_integers:
        s, i = _torch_integers[dtype_to]
        return quantize(np.where(receiver < 0, np.ceil(receiver), np.floor(receiver)), s, i, 0)
    return receiver


# ---------------------------------------------------------------------------
# Reductions: Tensor.sum/mean/prod/amax/amin/all/any/max/min
# ---------------------------------------------------------------------------


def _make_reduction(np_func: Callable) -> Callable:
    def fn(receiver: Any, *args: Any, **kwargs: Any) -> Any:
        dim = kwargs.get('dim', args[0] if args else None)
        keepdim = kwargs.get('keepdim', False)
        if dim is None:
            return np_func(receiver)
        return np_func(receiver, axis=dim, keepdims=keepdim)

    return fn


_method('sum')(_make_reduction(np.sum))
_method('mean')(_make_reduction(np.mean))
_method('prod')(_make_reduction(np.prod))
_method('amax', 'max')(_make_reduction(np.amax))
_method('amin', 'min')(_make_reduction(np.amin))
_method('all')(_make_reduction(np.all))
_method('any')(_make_reduction(np.any))


@_method('argmax')
def _argmax(receiver: Any, *args: Any, **kwargs: Any) -> Any:
    return np.argmax(receiver, axis=kwargs.get('dim', args[0] if args else None))


@_method('argmin')
def _argmin(receiver: Any, *args: Any, **kwargs: Any) -> Any:
    return np.argmin(receiver, axis=kwargs.get('dim', args[0] if args else None))


@_method('clamp', 'clip')
def _clamp(receiver: Any, *args: Any, **kwargs: Any) -> Any:
    lo = args[0] if len(args) > 0 else kwargs.get('min', None)
    hi = args[1] if len(args) > 1 else kwargs.get('max', None)
    return np.clip(receiver, lo, hi)
