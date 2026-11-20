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
    """Store the sparse ratings and the number of retained input rows.

    Attributes
    ----------
    ratings : scipy.sparse.csr_matrix
        Matrix of shape (users, games). Stored values are observed ratings.
    interactions : int
        Number of CSV rows read after filtering, before duplicate coordinates sum.
    """

    ratings: sparse.csr_matrix
    interactions: int


def load_interactions(
    path: Path,
    *,
    shape: tuple[int, int] | None = None,
    user_filter: Collection[int] | None = None,
) -> InteractionData:
    """Read processed user/item/rating rows into a sparse matrix.

    Parameters
    ----------
    path : Path
        CSV containing integer user_id/item_id and numeric rating columns.
    shape : tuple[int, int] or None
        Required (user count, game count), shared between splits. If omitted,
        infer dimensions from the largest IDs among retained rows.
    user_filter : collection[int] or None
        Keep only these users, retaining their original row IDs.

    Returns
    -------
    InteractionData
        CSR ratings and retained input-row count. Missing cells represent
        unobserved ratings. Duplicate coordinates are summed by SciPy, so callers
        must supply the deduplicated preprocessing output.

    Packed array.array buffers limit memory use. np.frombuffer creates views of
    these buffers, COO constructs indexed observations, and conversion to CSR
    supports efficient user-row slicing and matrix products.
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
    """Choose sorted user IDs without replacement.

    Parameters
    ----------
    total_users : int
        Number of available zero-based user IDs, 0 through total_users - 1.
    sample_size : int
        Number to retain, from 1 through total_users.
    seed : int
        NumPy generator seed, default 42, making repeated samples reproducible.

    Returns
    -------
    numpy.ndarray
        Sorted integer IDs of length sample_size.
    """

    if not 0 < sample_size <= total_users:
        raise ValueError("sample_size must be between 1 and total_users")
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(total_users, sample_size, replace=False))
