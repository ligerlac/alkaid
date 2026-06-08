from collections.abc import Sequence
from typing import TYPE_CHECKING

from .types import Op, QInterval

if TYPE_CHECKING:
    from .types import Op


def _s32(v: int) -> int:
    return ((int(v) & 0xFFFFFFFF) + 0x80000000) % 0x100000000 - 0x80000000


def _s64(v: int) -> int:
    v = int(v)
    if v > 0x7FFFFFFFFFFFFFFF:
        v -= 0x10000000000000000
    return v


def _op_from_v2_record(record: Sequence) -> Op:
    """Convert one v2 JSON op record to the v3 tuple-address format."""

    id0, id1, opcode, packed_data = int(record[0]), int(record[1]), int(record[2]), _s64(int(record[3]))
    qint = QInterval(*record[4])
    latency, cost = record[5], record[6]

    match opcode:
        case -2:
            op = Op((id0,), opcode, (), qint, latency, cost)
        case -1:
            op = Op((), opcode, (id0,), qint, latency, cost)
        case 0 | 1:
            op = Op((id0, id1), opcode, (packed_data,), qint, latency, cost)
        case 2 | 3:
            op = Op((id0,), opcode, (), qint, latency, cost)
        case 4:
            op = Op((id0,), opcode, (_s32(packed_data), _s32(packed_data >> 32)), qint, latency, cost)
        case 5:
            op = Op((), opcode, (packed_data,), qint, latency, cost)
        case 6:
            op = Op((id0, id1, _s32(packed_data)), opcode, (_s32(packed_data >> 32),), qint, latency, cost)
        case 7:
            op = Op((id0, id1), opcode, (), qint, latency, cost)
        case 8:
            op = Op((id0,), opcode, (packed_data,), qint, latency, cost)
        case 9:
            op = Op((id0,), opcode, (packed_data,), qint, latency, cost)
        case 10:
            op = Op((id0, id1), opcode, (_s32(packed_data), (packed_data >> 56) & 0xFF), qint, latency, cost)
        case _:
            raise ValueError(f'Unknown v2 opcode {opcode}')
    return op
