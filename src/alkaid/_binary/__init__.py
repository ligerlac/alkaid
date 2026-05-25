import struct

import numpy as np
from numpy.typing import NDArray

from .cmvm_bin import (
    cmvm_solve,
    csd_decompose,
    get_lsb_loc,
    iceil_log2,
    int_arr_to_csd,
    kernel_decompose,
    minimal_kif_batch,
    minimal_kif_scalar,
    overlap_counts,
    scm_solve,
)


def alir_interp_run(bin_logic: bytes, data: NDArray, n_threads: int = 1, dump: bool = False, ignore_lookup_oob: bool = False):
    from .alir_bin import run_interp

    if len(bin_logic) < 24:
        raise ValueError('Invalid binary logic data')
    magic, _spec, inp_size, _n_out, _n_ops, _n_tables = struct.unpack_from('<4sIIIII', bin_logic)
    if magic != b'ALIR':
        raise ValueError(f'Invalid ALIR bytecode magic {magic!r}')
    assert data.size % inp_size == 0, f'Input size {data.size} is not divisible by {inp_size}'

    inputs = np.ascontiguousarray(np.ravel(data), dtype=np.float64)
    return run_interp(bin_logic, inputs, n_threads, dump=dump, ignore_lookup_oob=ignore_lookup_oob)


def alir_interp_run_json_file(path: str, data: NDArray, n_threads: int = 1, dump: bool = False, ignore_lookup_oob: bool = False):
    from .alir_bin import run_interp_json_file

    inputs = np.ascontiguousarray(np.ravel(data), dtype=np.float64)
    return run_interp_json_file(path, inputs, n_threads, dump=dump, ignore_lookup_oob=ignore_lookup_oob)


__all__ = [
    'alir_interp_run',
    'alir_interp_run_json_file',
    'int_arr_to_csd',
    'csd_decompose',
    'get_lsb_loc',
    'kernel_decompose',
    'minimal_kif_batch',
    'minimal_kif_scalar',
    'scm_solve',
    'cmvm_solve',
    'iceil_log2',
    'overlap_counts',
]
