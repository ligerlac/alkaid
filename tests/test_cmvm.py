import numpy as np
import pytest

from alkaid._binary import cmvm_solve, csd_decompose, kernel_decompose
from alkaid.trace.passes import _fast_optimize
from alkaid.types import Pipeline


@pytest.fixture(params=[2, 4, 8, 12])
def n_dim(request) -> int:
    return request.param


@pytest.fixture(params=[2, 4, 8])
def bits(request) -> int:
    return request.param


@pytest.fixture
def kernel(n_dim, bits):
    kernel = np.round((np.random.rand(n_dim, n_dim) - 0.5) * 2 ** (bits + 1)).astype(np.float32)
    return kernel


def test_decompose(kernel):
    csd, shift0, shift1 = csd_decompose(kernel.astype(np.float32))
    shift2 = np.arange(csd.shape[-1])
    recon = csd * (2.0 ** shift0[:, None, None]) * (2.0 ** shift1[None, :, None]) * (2.0 ** shift2[None, None, :])
    recon_sum = np.sum(recon, axis=-1)
    assert np.all(recon_sum == kernel)


@pytest.mark.parametrize('dc', [-2, -1, 0, 1, 2])
def test_kernel_decompose(kernel, dc: int):
    m0, m1 = kernel_decompose(kernel.astype(np.float32), dc=dc)
    recon = m0 @ m1
    assert np.all(recon == kernel)


@pytest.mark.parametrize('hard_dc', [0, 2, -1])
@pytest.mark.parametrize('method0', ['mc', 'wmc'])
@pytest.mark.parametrize('method1', ['mc', 'wmc'])
@pytest.mark.parametrize('decompose_dc', [0, -1, -2])
@pytest.mark.parametrize('search_all_decompose_dc', [False, True])
def test_cmvm_solve(kernel, method0, method1, hard_dc, decompose_dc, search_all_decompose_dc):
    sol: Pipeline = cmvm_solve(
        kernel,
        hard_dc=hard_dc,
        method0=method0,
        method1=method1,
        decompose_dc=decompose_dc,
        search_all_decompose_dc=search_all_decompose_dc,
    )

    combs = tuple(_fast_optimize(stage, False) for stage in sol.solutions)
    pipe = Pipeline(combs)
    _ = pipe.__repr__()

    np.testing.assert_allclose(pipe.kernel, kernel)


def test_cmvm_output_uses_tuple_ops():
    sol: Pipeline = cmvm_solve(np.array([[1, -2], [3, 4]], dtype=np.float32), hard_dc=0)
    ops = [op for stage in sol.solutions for op in stage.ops]

    assert ops
    assert all(op.opcode in (-1, 0, 1) for op in ops)
    assert all(isinstance(op.addr, tuple) and isinstance(op.data, tuple) for op in ops)
    assert all(not hasattr(op, name) for op in ops for name in ('id0', 'id1'))
    assert all(
        (op.addr == () and len(op.data) == 1) if op.opcode == -1 else (len(op.addr) == 2 and len(op.data) == 1) for op in ops
    )
