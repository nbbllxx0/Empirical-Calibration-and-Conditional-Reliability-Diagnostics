from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15

MX_DOUBLE = 6
MX_SINGLE = 7
MX_INT8 = 8
MX_UINT8 = 9
MX_INT16 = 10
MX_UINT16 = 11
MX_INT32 = 12
MX_UINT32 = 13
MX_INT64 = 14
MX_UINT64 = 15

DTYPE_BY_MI = {
    MI_INT8: np.int8,
    MI_UINT8: np.uint8,
    MI_INT16: np.int16,
    MI_UINT16: np.uint16,
    MI_INT32: np.int32,
    MI_UINT32: np.uint32,
    MI_SINGLE: np.float32,
    MI_DOUBLE: np.float64,
    MI_INT64: np.int64,
    MI_UINT64: np.uint64,
}


def _align8(size: int) -> int:
    return size + ((8 - size % 8) % 8)


def _read_tag(buf: bytes, offset: int, endian: str) -> tuple[int, int, int, bytes | None]:
    word = struct.unpack_from(endian + "I", buf, offset)[0]
    small_size = word >> 16
    small_type = word & 0xFFFF
    if small_size:
        data = buf[offset + 4 : offset + 4 + small_size]
        return small_type, small_size, offset + 8, data
    dtype, size = struct.unpack_from(endian + "II", buf, offset)
    return dtype, size, offset + 8, None


def _read_element(buf: bytes, offset: int, endian: str) -> tuple[int, bytes, int]:
    dtype, size, data_offset, inline = _read_tag(buf, offset, endian)
    if inline is not None:
        return dtype, inline, data_offset
    data = buf[data_offset : data_offset + size]
    if dtype == MI_COMPRESSED:
        return dtype, data, data_offset + size
    return dtype, data, data_offset + _align8(size)


def _to_ints(data: bytes, dtype: int, endian: str) -> list[int]:
    np_dtype = np.dtype(DTYPE_BY_MI.get(dtype, np.uint32)).newbyteorder(endian)
    return [int(v) for v in np.frombuffer(data, dtype=np_dtype)]


def _matrix_from_payload(payload: bytes, endian: str) -> tuple[str, np.ndarray] | None:
    offset = 0
    _, flags_data, offset = _read_element(payload, offset, endian)
    if len(flags_data) < 8:
        return None
    class_bits = struct.unpack_from(endian + "I", flags_data, 0)[0]
    mx_class = class_bits & 0xFF
    if mx_class not in {MX_DOUBLE, MX_SINGLE, MX_INT8, MX_UINT8, MX_INT16, MX_UINT16, MX_INT32, MX_UINT32, MX_INT64, MX_UINT64}:
        return None
    dim_type, dim_data, offset = _read_element(payload, offset, endian)
    dims = _to_ints(dim_data, dim_type, endian)
    if not dims:
        return None
    _, name_data, offset = _read_element(payload, offset, endian)
    name = name_data.decode("latin1", errors="ignore")
    real_type, real_data, _ = _read_element(payload, offset, endian)
    if real_type not in DTYPE_BY_MI:
        return None
    dtype = np.dtype(DTYPE_BY_MI[real_type]).newbyteorder(endian)
    arr = np.frombuffer(real_data, dtype=dtype).copy()
    expected = int(np.prod(dims))
    if arr.size < expected:
        return None
    arr = arr[:expected].reshape(tuple(dims), order="F")
    return name, arr


def loadmat_numeric(path: str | Path) -> dict[str, np.ndarray]:
    data = Path(path).read_bytes()
    if len(data) < 128:
        raise ValueError(f"Not a MATLAB v5 file: {path}")
    endian_indicator = data[126:128]
    endian = "<" if endian_indicator == b"IM" else ">"
    offset = 128
    variables: dict[str, np.ndarray] = {}
    while offset + 8 <= len(data):
        dtype, payload, offset = _read_element(data, offset, endian)
        if dtype == MI_COMPRESSED:
            decompressed = zlib.decompress(payload)
            inner_offset = 0
            while inner_offset + 8 <= len(decompressed):
                inner_type, inner_payload, inner_offset = _read_element(decompressed, inner_offset, endian)
                if inner_type == MI_MATRIX:
                    parsed = _matrix_from_payload(inner_payload, endian)
                    if parsed:
                        variables[parsed[0]] = parsed[1]
        elif dtype == MI_MATRIX:
            parsed = _matrix_from_payload(payload, endian)
            if parsed:
                variables[parsed[0]] = parsed[1]
        else:
            break
    return variables


def loadmat_first_numeric(path: str | Path) -> np.ndarray:
    variables = loadmat_numeric(path)
    candidates = [v for k, v in variables.items() if not k.startswith("__") and np.asarray(v).size > 0]
    if not candidates:
        raise ValueError(f"No numeric arrays found in {path}")
    candidates.sort(key=lambda arr: arr.size, reverse=True)
    arr = np.asarray(candidates[0])
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.astype(np.float32, copy=False)
