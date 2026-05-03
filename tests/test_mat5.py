import struct
import zlib
from pathlib import Path

import numpy as np

from bearing_dt.data.mat5 import (
    MI_COMPRESSED,
    MI_DOUBLE,
    MI_INT8,
    MI_INT32,
    MI_MATRIX,
    MI_UINT32,
    MX_DOUBLE,
    loadmat_numeric,
)


def _align8(size: int) -> int:
    return size + ((8 - size % 8) % 8)


def _element(dtype: int, payload: bytes) -> bytes:
    return struct.pack("<II", dtype, len(payload)) + payload + (b"\x00" * (_align8(len(payload)) - len(payload)))


def _matrix(name: str, values: list[float]) -> bytes:
    flags = _element(MI_UINT32, struct.pack("<II", MX_DOUBLE, 0))
    dims = _element(MI_INT32, struct.pack("<II", len(values), 1))
    var_name = _element(MI_INT8, name.encode("ascii"))
    real = _element(MI_DOUBLE, struct.pack("<" + "d" * len(values), *values))
    payload = flags + dims + var_name + real
    return _element(MI_MATRIX, payload)


def _compressed_matrix(name: str, values: list[float]) -> bytes:
    compressed = zlib.compress(_matrix(name, values))
    if len(compressed) % 8 == 0:
        compressed += b"\x00"
    return struct.pack("<II", MI_COMPRESSED, len(compressed)) + compressed


def test_loadmat_reads_unpadded_back_to_back_compressed_elements(tmp_path: Path):
    header = bytearray(128)
    text = b"MATLAB 5.0 MAT-file"
    header[: len(text)] = text
    header[124:126] = struct.pack("<H", 0x0100)
    header[126:128] = b"IM"

    path = tmp_path / "unpadded_compressed.mat"
    path.write_bytes(bytes(header) + _compressed_matrix("a", [1.0, 2.0]) + _compressed_matrix("b", [3.0, 4.0]))

    variables = loadmat_numeric(path)

    assert set(variables) == {"a", "b"}
    np.testing.assert_allclose(variables["a"].reshape(-1), [1.0, 2.0])
    np.testing.assert_allclose(variables["b"].reshape(-1), [3.0, 4.0])
