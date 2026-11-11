"""Sparse interaction data loading utilities.

The processed CSV files are much larger than the available item catalog.  This
module therefore stores ratings in a SciPy sparse matrix instead of building a
dense ``users x games`` table.  Missing cells mean "not rated"; they are not
zero-star ratings.
"""

from __future__ import annotations

import csv
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class InteractionData:
    """A sparse rating matrix and the number of CSV rows used to build it."""

    ratings: sparse.csr_matrix
    interactions: int


def load_interactions(
    path: Path,
    *,
    shape: tuple[int, int] | None = None,
    user_filter: Collection[int] | None = None,
) -> InteractionData:
    """Load ``user_id,item_id,rating`` CSV rows into a CSR matrix.

    ``array.array`` is used instead of Python lists.  Twelve million Python
    integer objects would consume far more memory than packed 32-bit arrays.
    A user filter supports reproducible, representative smoke experiments: the
    whole CSV is scanned, so selected users retain ratings across all games.
    """

    selected = set(user_filter) if user_filter is not None else None
    users = array("I")
    items = array("I")
    values = array("f")
    max_user = -1
    max_item = -1

    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected = {"user_id", "item_id", "rating"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"Expected columns {sorted(expected)} in {path}")
        for row in reader:
            user_id = int(row["user_id"])
            if selected is not None and user_id not in selected:
                continue
            item_id = int(row["item_id"])
            users.append(user_id)
            items.append(item_id)
            values.append(float(row["rating"]))
            max_user = max(max_user, user_id)
            max_item = max(max_item, item_id)

    if shape is None:
        shape = (max_user + 1, max_item + 1)
    if max_user >= shape[0] or max_item >= shape[1]:
        raise ValueError(f"Row IDs exceed requested matrix shape {shape}")

    # np.frombuffer creates zero-copy views over the packed arrays.
    row_ids = np.frombuffer(users, dtype=np.uint32)
    column_ids = np.frombuffer(items, dtype=np.uint32)
    ratings = np.frombuffer(values, dtype=np.float32)
    matrix = sparse.coo_matrix((ratings, (row_ids, column_ids)), shape=shape).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return InteractionData(matrix, len(values))


def sample_user_ids(total_users: int, sample_size: int, seed: int = 42) -> np.ndarray:
    """Return a sorted deterministic user sample without replacement."""

    if not 0 < sample_size <= total_users:
        raise ValueError("sample_size must be between 1 and total_users")
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(total_users, sample_size, replace=False))
