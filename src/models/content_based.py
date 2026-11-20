"""Content-based recommendations from descriptions, categories, and mechanics."""

from __future__ import annotations

import ast
import csv
import html
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from scipy import sparse


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9'-]+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "another",
    "around", "because", "before", "being", "between", "board", "both",
    "can", "cards", "during", "each", "from", "game", "games", "have",
    "into", "more", "most", "other", "over", "player", "players", "than",
    "that", "their", "them", "then", "there", "these", "they", "this",
    "through", "under", "using", "when", "where", "which", "while", "will",
    "with", "within", "would", "your",
}


@dataclass(frozen=True)
class ContentConfig:
    """Feature and rating-estimate settings for content recommendation.

    Args:
        max_description_features: Maximum number of selected description terms.
        min_document_frequency: Minimum games containing a selected term.
        description_weight: Weight in [0, 1] for normalized TF-IDF; normalized
            category/mechanic tags receive 1 - description_weight.
        liked_threshold: Minimum training rating used to build a user profile.
        rating_scale: Nonnegative similarity-to-rating multiplier for RMSE only.
    Settings are frozen and compared using validation data in the experiment."""

    max_description_features: int = 4_000
    min_document_frequency: int = 2
    description_weight: float = 0.5
    liked_threshold: float = 8.0
    rating_scale: float = 2.0

    def __post_init__(self) -> None:
        """Validate feature limits, mixture weight, liking threshold and rating scale.

        Called automatically by the dataclass. Returns None; invalid limits or
        weights raise ValueError before the configuration reaches a model."""

        if self.max_description_features <= 0:
            raise ValueError("max_description_features must be positive")
        if self.min_document_frequency <= 0:
            raise ValueError("min_document_frequency must be positive")
        if not 0.0 <= self.description_weight <= 1.0:
            raise ValueError("description_weight must be between 0 and 1")
        if not 1.0 <= self.liked_threshold <= 10.0:
            raise ValueError("liked_threshold must be between 1 and 10")
        if self.rating_scale < 0:
            raise ValueError("rating_scale must be non-negative")


@dataclass(frozen=True)
class ItemFeatures:
    """Normalized item features with their description and tag dimensions.

    Args:
        matrix: CSR matrix with games on rows and combined features on columns.
        description_features: Number of selected TF-IDF term columns.
        tag_features: Number of category/mechanic columns.
    The frozen object keeps feature metadata attached to the corresponding matrix."""

    matrix: sparse.csr_matrix
    description_features: int
    tag_features: int


def build_item_features(
    games_path: Path,
    config: ContentConfig,
    *,
    expected_items: int | None = None,
) -> ItemFeatures:
    """Build normalized game vectors from descriptions and category/mechanic tags.

    Args:
        games_path: Metadata CSV containing contiguous item_id values and
            description, categories and mechanics fields.
        config: ContentConfig governing vocabulary limits and mixture weights.
        expected_items: Optional required row count, usually the training width.
    Returns:
        ItemFeatures with TF-IDF and tag blocks concatenated and L2-normalized.
        Vocabulary fitting uses the available catalog text, not held-out ratings.
    Raises:
        ValueError: If IDs/columns/count are inconsistent or no features remain."""

    # Step 1: load games in contiguous item-ID order so feature rows line up
    # exactly with the interaction-matrix columns.
    rows = _load_games(games_path)
    if expected_items is not None and len(rows) != expected_items:
        raise ValueError(
            f"Expected {expected_items} games but found {len(rows)} in {games_path}"
        )
    # Step 2: create independent normalized description and tag representations.
    descriptions = [_token_counts(row.get("description", "")) for row in rows]
    description_matrix = _tfidf_matrix(
        descriptions,
        max_features=config.max_description_features,
        min_document_frequency=config.min_document_frequency,
    )
    tag_matrix = _tag_matrix(rows)

    # Step 3: apply the validation-selected weights and normalize the combined
    # vector so its dot product is cosine similarity.
    weighted_parts: list[sparse.csr_matrix] = []
    if config.description_weight > 0 and description_matrix.shape[1]:
        weighted_parts.append(description_matrix * config.description_weight)
    tag_weight = 1.0 - config.description_weight
    if tag_weight > 0 and tag_matrix.shape[1]:
        weighted_parts.append(tag_matrix * tag_weight)
    if not weighted_parts:
        raise ValueError("No content features were produced from the games file")

    combined = sparse.hstack(weighted_parts, format="csr", dtype=np.float32)
    combined.eliminate_zeros()
    return ItemFeatures(
        matrix=_normalize_rows(combined),
        description_features=description_matrix.shape[1],
        tag_features=tag_matrix.shape[1],
    )


class ContentBasedRecommender:
    """Rank games by cosine similarity to a user's liked-game content profile.

    Training ratings at least liked_threshold contribute weight
    rating - liked_threshold + 1 to the profile. Profile and item vectors are
    L2-normalized, so their dot product gives the content score. predict_pairs()
    uses a separate user-mean-centered transformation solely to report RMSE."""

    def __init__(self, config: ContentConfig | None = None) -> None:
        """Create an unfitted content recommender.

        Args:
            config: Immutable feature/profile settings; None selects ContentConfig().
        Returns:
            None. fit() constructs user profiles and rating baselines."""

        self.config = config or ContentConfig()
        self.train_: sparse.csr_matrix | None = None
        self.item_features_: sparse.csr_matrix | None = None
        self.user_profiles_: sparse.csr_matrix | None = None
        self.user_means_: np.ndarray | None = None
        self.catalog_similarity_: np.ndarray | None = None

    def fit(
        self,
        train: sparse.csr_matrix,
        item_features: sparse.csr_matrix,
    ) -> "ContentBasedRecommender":
        """Aggregate liked training games into normalized user preference vectors.

        Args:
            train: Sparse user-by-item matrix containing observed training ratings.
            item_features: Sparse item-by-feature matrix from build_item_features;
                its row order must match the item columns of train.
        Returns:
            This fitted model. A user without a liked game gets a zero profile;
            their content similarities are zero rather than evidence of dislike.
        Raises:
            ValueError: If item dimensions differ or training has no ratings."""

        train = train.tocsr().astype(np.float32)
        item_features = item_features.tocsr().astype(np.float32)
        if train.shape[1] != item_features.shape[0]:
            raise ValueError("train items and item feature rows must match")
        if train.nnz == 0:
            raise ValueError("train must contain at least one rating")

        # Step 1: keep liked train games only and give higher ratings more weight.
        liked = train.copy()
        liked.data = np.where(
            liked.data >= self.config.liked_threshold,
            liked.data - self.config.liked_threshold + 1.0,
            0.0,
        ).astype(np.float32)
        liked.eliminate_zeros()
        # Step 2: aggregate liked-game features into one normalized user profile.
        profiles = _normalize_rows((liked @ item_features).tocsr())

        # Step 3: retain user rating baselines so cosine similarity can also be
        # converted to a bounded held-out rating estimate for RMSE reporting.
        counts = np.asarray(train.getnnz(axis=1), dtype=np.float32)
        sums = np.asarray(train.sum(axis=1)).ravel().astype(np.float32)
        global_mean = float(train.data.mean())
        user_means = np.divide(
            sums,
            counts,
            out=np.full(train.shape[0], global_mean, dtype=np.float32),
            where=counts > 0,
        )
        mean_item = np.asarray(item_features.mean(axis=0)).ravel()

        self.train_ = train
        self.item_features_ = item_features
        self.user_profiles_ = profiles
        self.user_means_ = user_means
        self.catalog_similarity_ = np.asarray(profiles @ mean_item).ravel()
        return self

    def predict_pairs(
        self,
        user_ids: Iterable[int],
        item_ids: Iterable[int],
        *,
        batch_size: int = 50_000,
    ) -> np.ndarray:
        """Convert content similarities into bounded estimates for held-out RMSE.

        The estimate is user_mean + rating_scale * (pair_similarity - the
        user's mean catalog similarity), clipped to [1, 10]. Ranking uses raw
        cosine scores instead of this rating conversion.

        Args:
            user_ids: Iterable of valid zero-based fitted user rows.
            item_ids: Equally long iterable of valid fitted game columns.
            batch_size: Positive maximum number of pairs scored together.
        Returns:
            Float32 array of one estimate per pair, or an empty array for no pairs.
        Raises:
            ValueError: If the two ID sequences have different lengths."""

        self._require_fit()
        users = np.asarray(list(user_ids), dtype=np.int64)
        items = np.asarray(list(item_ids), dtype=np.int64)
        if users.shape != items.shape:
            raise ValueError("user_ids and item_ids must have the same length")
        if users.size == 0:
            return np.empty(0, dtype=np.float32)
        assert self.user_profiles_ is not None
        assert self.item_features_ is not None
        assert self.user_means_ is not None
        assert self.catalog_similarity_ is not None

        # Step 1: compute only requested user-item similarities in bounded batches.
        similarities = np.empty(users.size, dtype=np.float32)
        for start in range(0, users.size, batch_size):
            stop = min(start + batch_size, users.size)
            profile_rows = self.user_profiles_[users[start:stop]]
            item_rows = self.item_features_[items[start:stop]]
            similarities[start:stop] = np.asarray(
                profile_rows.multiply(item_rows).sum(axis=1)
            ).ravel()
        # Step 2: center similarity by each profile's catalog-average similarity,
        # add the user's rating baseline, and clip to the original 1-10 range.
        predictions = self.user_means_[users] + self.config.rating_scale * (
            similarities - self.catalog_similarity_[users]
        )
        return np.clip(predictions, 1.0, 10.0).astype(np.float32)

    def iter_recommendations(
        self,
        user_ids: Iterable[int],
        *,
        top_k: int = 10,
        batch_size: int = 256,
    ) -> Iterator[tuple[int, list[tuple[int, float]]]]:
        """Yield unseen-game rankings in bounded user batches.

        Args:
            user_ids: Iterable of valid zero-based fitted user rows.
            top_k: Positive maximum number of games returned per user.
            batch_size: Positive maximum number of user profiles scored together.
        Returns:
            Iterator of (user_id, [(item_id, cosine_score), ...]). Train-rated
            games are masked. Selected games are sorted by score, then item ID.
        Raises:
            ValueError: If top_k is nonpositive."""

        self._require_fit()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        users = np.asarray(list(user_ids), dtype=np.int64)
        assert self.train_ is not None
        assert self.item_features_ is not None
        assert self.user_profiles_ is not None

        item_ids = np.arange(self.train_.shape[1])
        for start in range(0, users.size, batch_size):
            # Step 1: score a bounded user batch against the complete catalog.
            batch_users = users[start:start + batch_size]
            scores = (self.user_profiles_[batch_users] @ self.item_features_.T).toarray()
            for local_index, user_id in enumerate(batch_users):
                user_scores = scores[local_index]
                # Step 2: mask train-seen games, then select and deterministically
                # order the highest-scoring unseen games.
                user_scores[self.train_.getrow(int(user_id)).indices] = -np.inf
                available = np.flatnonzero(np.isfinite(user_scores))
                keep = min(top_k, available.size)
                if available.size > keep:
                    available = available[
                        np.argpartition(user_scores[available], -keep)[-keep:]
                    ]
                order = np.lexsort((item_ids[available], -user_scores[available]))
                ranked = available[order]
                yield int(user_id), [
                    (int(item_id), float(user_scores[item_id])) for item_id in ranked
                ]

    def recommend(
        self,
        user_ids: Iterable[int],
        *,
        top_k: int = 10,
        batch_size: int = 256,
    ) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
        """Collect the batched iterator into recommendation and score dictionaries.

        Args:
            user_ids: Iterable of valid zero-based fitted user IDs.
            top_k: Positive maximum number of results per user.
            batch_size: Positive number of profiles per scoring batch.
        Returns:
            (recommendations, scores), each keyed by user ID. Values are aligned
            lists of unseen item IDs and their raw cosine similarities."""

        recommendations: dict[int, list[int]] = {}
        scores: dict[int, list[float]] = {}
        for user_id, ranking in self.iter_recommendations(
            user_ids, top_k=top_k, batch_size=batch_size
        ):
            recommendations[user_id] = [item_id for item_id, _ in ranking]
            scores[user_id] = [score for _, score in ranking]
        return recommendations, scores

    def _require_fit(self) -> None:
        """Raise RuntimeError if training data or user profiles are absent.

        Returns:
            None when fit() has populated the state used by scoring methods."""

        if self.train_ is None or self.user_profiles_ is None:
            raise RuntimeError("model must be fitted before prediction")


def _load_games(path: Path) -> list[dict[str, str]]:
    """Read metadata in contiguous item-ID order and validate required fields.

    Args:
        path: UTF-8 games CSV with item_id, description, categories and mechanics.
    Returns:
        Metadata rows sorted by integer item_id from zero to item count minus one.
    Raises:
        ValueError: If rows/required columns are missing or IDs are not contiguous."""

    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"item_id", "description", "categories", "mechanics"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Expected columns {sorted(required)} in {path}")
    rows.sort(key=lambda row: int(row["item_id"]))
    item_ids = [int(row["item_id"]) for row in rows]
    if item_ids != list(range(len(rows))):
        raise ValueError("games item_id values must be contiguous and zero-based")
    return rows


def _token_counts(text: str) -> Counter[str]:
    """Decode HTML, lowercase text, and count retained description tokens.

    Args:
        text: One game description, possibly containing HTML entities or tags.
    Returns:
        Counter of matching tokens at least three characters long, excluding
        STOPWORDS. TOKEN_PATTERN defines the accepted letters/digits/apostrophes."""

    cleaned = HTML_TAG_PATTERN.sub(" ", html.unescape(text)).lower()
    return Counter(
        token
        for token in TOKEN_PATTERN.findall(cleaned)
        if token not in STOPWORDS and len(token) >= 3
    )


def _tfidf_matrix(
    documents: list[Counter[str]],
    *,
    max_features: int,
    min_document_frequency: int,
) -> sparse.csr_matrix:
    """Calculate sparse, row-normalized sublinear TF-IDF description vectors.

    A retained token has tf = 1 + log(count) and
    idf = log((1 + number_of_documents)/(1 + document_frequency)) + 1.
    Terms pass the document-frequency floor, then the most frequent corpus
    terms are kept; lexicographic order resolves equal corpus frequencies.

    Args:
        documents: One token Counter per catalog game, in item-ID order.
        max_features: Positive vocabulary cap.
        min_document_frequency: Minimum number of documents containing a token.
    Returns:
        Float32 CSR matrix with one row per document and at most max_features
        columns. If no token survives, its shape is (documents, 0)."""

    document_frequency: Counter[str] = Counter()
    corpus_frequency: Counter[str] = Counter()
    for counts in documents:
        document_frequency.update(counts.keys())
        corpus_frequency.update(counts)
    candidates = [
        token
        for token, count in document_frequency.items()
        if count >= min_document_frequency
    ]
    candidates.sort(key=lambda token: (-corpus_frequency[token], token))
    vocabulary = {token: index for index, token in enumerate(candidates[:max_features])}
    if not vocabulary:
        return sparse.csr_matrix((len(documents), 0), dtype=np.float32)

    row_ids: list[int] = []
    column_ids: list[int] = []
    values: list[float] = []
    document_count = len(documents)
    for row_id, counts in enumerate(documents):
        for token, count in counts.items():
            column_id = vocabulary.get(token)
            if column_id is None:
                continue
            inverse_document_frequency = math.log(
                (1 + document_count) / (1 + document_frequency[token])
            ) + 1.0
            row_ids.append(row_id)
            column_ids.append(column_id)
            values.append((1.0 + math.log(count)) * inverse_document_frequency)
    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (row_ids, column_ids)),
        shape=(document_count, len(vocabulary)),
    )
    return _normalize_rows(matrix)


def _tag_matrix(rows: list[dict[str, str]]) -> sparse.csr_matrix:
    """Encode category and mechanic lists as separate normalized multi-hot features.

    Args:
        rows: Metadata rows containing serialized categories and mechanics lists.
    Returns:
        Float32 game-by-tag CSR matrix. Prefixes keep an identically named
        category and mechanic distinct; sorted tags make columns reproducible."""

    per_item: list[list[str]] = []
    all_tags: set[str] = set()
    for row in rows:
        tags = [f"category:{value}" for value in _parse_list(row["categories"])]
        tags.extend(f"mechanic:{value}" for value in _parse_list(row["mechanics"]))
        unique = sorted(set(tags))
        per_item.append(unique)
        all_tags.update(unique)
    vocabulary = {tag: index for index, tag in enumerate(sorted(all_tags))}
    row_ids: list[int] = []
    column_ids: list[int] = []
    for row_id, tags in enumerate(per_item):
        row_ids.extend([row_id] * len(tags))
        column_ids.extend(vocabulary[tag] for tag in tags)
    values = np.ones(len(row_ids), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (values, (row_ids, column_ids)),
        shape=(len(rows), len(vocabulary)),
    )
    return _normalize_rows(matrix)


def _parse_list(value: str) -> list[str]:
    """Safely parse a CSV-serialized Python list of category/mechanic labels.

    Args:
        value: Text such as a bracketed list of quoted labels.
    Returns:
        Nonempty stripped label strings. Malformed or non-list values give [].
        ast.literal_eval accepts literals without executing arbitrary Python."""

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _normalize_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """L2-normalize nonzero sparse feature rows while preserving empty rows.

    Args:
        matrix: Sparse row-by-feature matrix.
    Returns:
        Float32 CSR matrix where nonempty rows have unit Euclidean norm.
        A zero norm receives inverse zero to avoid division by zero."""

    matrix = matrix.tocsr().astype(np.float32)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.divide(
        1.0,
        norms,
        out=np.zeros_like(norms, dtype=np.float32),
        where=norms > 0,
    )
    return (sparse.diags(inverse) @ matrix).tocsr()
