import numpy as np

from ...types import CombLogic, Op
from ..fixed_variable import LookupTable


def _index_remap(op: Op, idx_map: dict[int, int]) -> Op:
    if op.opcode == -1:
        return op
    addr = tuple(idx_map.get(i, i) for i in op.addr)
    return Op(addr, op.opcode, op.data, op.qint, op.latency, op.cost)


def remap_table_idxs(ops: list[Op], tables: tuple[LookupTable, ...] | None):
    if tables is None:
        return ops, tables
    if len({op.data[0] for op in ops if op.opcode == 8}) == len(tables):
        return ops, tables
    remap_index: dict[int, int] = {}
    new_tables: list[LookupTable] = []
    new_ops = ops.copy()

    for i, op in enumerate(ops):
        if op.opcode != 8:
            continue
        table_idx = op.data[0]
        if table_idx not in remap_index:
            remap_index[table_idx] = len(new_tables)
            new_tables.append(tables[table_idx])
        new_table_idx = remap_index[table_idx]
        new_ops[i] = Op(op.addr, op.opcode, (new_table_idx,), op.qint, op.latency, op.cost)

    return new_ops, tuple(new_tables)


def dead_code_elimin(comb: CombLogic, keep_dead_inputs=False) -> CombLogic:
    "dead code elimination"
    dead = np.ones(len(comb.ops), dtype=bool)
    for idx in comb.out_idxs:
        if idx < 0:
            continue
        dead[idx] = False

    for i in range(len(comb.ops) - 1, -1, -1):
        op = comb.ops[i]
        if dead[i] and not (keep_dead_inputs and op.opcode == -1):
            continue
        for idx in op.input_ids:
            dead[idx] = False

    new_idxs = np.cumsum(~dead) - 1
    idx_map = {int(i): int(new_idxs[i]) for i in range(len(comb.ops))}
    new_ops = [_index_remap(op, idx_map) for i, op in enumerate(comb.ops) if not dead[i]]
    new_out_idxs = [idx_map[idx] if idx >= 0 else -1 for idx in comb.out_idxs]

    new_ops, new_tables = remap_table_idxs(new_ops, comb.lookup_tables)

    return CombLogic(
        comb.shape,
        comb.inp_shifts,
        new_out_idxs,
        comb.out_shifts,
        comb.out_negs,
        new_ops,
        comb.carry_size,
        comb.adder_size,
        new_tables,
    )
