from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


Rows = list[dict[str, Any]]


def _parse_value(value: str) -> Any:
    if value == "":
        return ""
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def read_rows_csv(path: str | Path) -> Rows:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{k: _parse_value(v) for k, v in row.items()} for row in reader]


def write_rows_csv(path: str | Path, rows: Rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def column(rows: Rows, name: str) -> list[Any]:
    return [row[name] for row in rows]


def unique_values(rows: Rows, name: str) -> list[Any]:
    return sorted({row[name] for row in rows})


def group_by(rows: Rows, name: str) -> dict[Any, Rows]:
    groups: dict[Any, Rows] = defaultdict(list)
    for row in rows:
        groups[row[name]].append(row)
    return dict(groups)


def indices_where(rows: Rows, predicate: Callable[[dict[str, Any]], bool]) -> list[int]:
    return [idx for idx, row in enumerate(rows) if predicate(row)]


def take(rows: Rows, indices: Iterable[int]) -> Rows:
    return [rows[int(i)] for i in indices]
