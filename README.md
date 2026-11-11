# Board Game Recommendation System

This machine learning project builds personalized board game recommendations from BoardGameGeek (BGG) user ratings.

## Dataset Selection

This project now uses the **2025 v4 snapshot** of
`jvanelteren/boardgamegeek-reviews`.

- 26,200,012 raw user-game ratings
- 27,780 game metadata rows
- Metadata including player count, playtime, complexity, categories, and mechanics
- Suitable for comparing collaborative filtering and content-based recommendation on the same dataset

The v4 archive is pinned by dataset version and checksum. Older v3 data was
removed after the v4 outputs were verified; all current preprocessing and model
results use v4 only. The official BGG rank CSV contains only aggregated
game-level scores and cannot replace the user-game interactions required for
personalized recommendations.

The original source is the [BoardGameGeek XML API](https://boardgamegeek.com/wiki/page/XML%20API), and the redistributed dataset is available through [Kaggle BoardGameGeek Reviews](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews). The BGG API terms should be reviewed before using the data.

## Preprocessing

The raw v4 dataset contains **26,200,012 ratings**. Preprocessing resolves
duplicate/conflicting user-game pairs on disk, joins the 2025 metadata, and
builds a deterministic common evaluation cohort.

| Raw data | Criterion | Processing |
| --- | --- | --- |
| Missing user or game IDs and invalid ratings | Valid BGG ratings range from 1 to 10 | Removed 12 invalid rows; decimal ratings remain unchanged |
| Exact duplicate or conflicting user-game rows | Keep one exact duplicate; reject a pair with different ratings | No duplicates or conflicting pairs were present in this snapshot |
| Missing metadata or too little training support | Use metadata-backed games with at least 100 provisional training ratings | Selected the 3,000 most-rated eligible games using train-only counts |
| Users with insufficient training history | Retain enough history for preference learning | Kept users with at least 10 training ratings |
| Raw IDs are strings with large value ranges | Standardize model inputs | Encoded user and game IDs as zero-based integers |
| Ranking users need a hidden liked game | Preserve at least one 8+ game in every split | Deterministic user-aware train/validation/test split with seed 42 |

The final processed dataset contains **248,457 users, 3,000 games, and
19,745,913 interactions**: 15,473,315 train, 2,138,126 validation, and
2,134,472 test. It has **97.35% sparsity** and 246,319 common ranking users.

```bash
.venv/bin/python -m src.data.download --dataset-version 4
unzip -j data/raw/v4/boardgamegeek-reviews-v4.zip \
  bgg-26m-reviews.csv games_detailed_info2025.csv -d data/raw/v4
.venv/bin/python -m src.data.preprocess
```

### Assignment experiment subset

The original processed dataset is preserved. A smaller experiment dataset is
selected using **training-split statistics only** to avoid validation/test
leakage:

- 2,000 games with the most training ratings
- users with at least 20 training ratings inside that catalog
- 20,000 users sampled proportionally from activity groups `20-39`, `40-79`,
  and `80+`
- compact zero-based IDs written separately under `data/experiment_v4/`

```bash
.venv/bin/python -m src.data.create_experiment_subset \
  --processed-dir data/processed/v4 \
  --output-dir data/experiment_v4 \
  --max-items 2000 --max-users 20000 \
  --min-user-train-ratings 20 --seed 42
```

The generated v4 subset contains **2,029,222 interactions**: 1,599,055 train,
215,504 validation, and 214,663 test interactions. It has 20,000 evaluation
users, 2,000 games, and 94.93% matrix sparsity. Generated CSV files remain
excluded from Git; the selection code, report schema, and reproduction command
are versioned.

![Rating counts per user and game](reports/eda/interaction_count_distributions.png)

![Rating distribution](reports/eda/rating_distribution.png)

## Project Structure

```text
data/                     Local datasets (excluded from Git)
reports/                  Model reports, metrics, and figures
src/data/                 Download, preprocessing, loading, and EDA
src/models/               Popularity, ItemCF, and Matrix Factorization
src/experiments/          Search, evaluation, outputs, and final tests
src/metrics.py            Metrics shared by every model
tests/                    Regression and model tests
```

## Modeling Roadmap

1. Use popularity-based recommendations as the baseline.
2. Implement collaborative filtering with Surprise SVD or implicit ALS.
3. Implement content-based recommendations using game categories, mechanics, and complexity.
4. Compare models using RMSE, Recall@10, NDCG@10, and coverage.

## Item-Item Collaborative Filtering Experiment

The first personalized model compares four methods from the recommendation
lecture: cosine, adjusted cosine, Pearson-style correlation, and Jaccard
similarity. It also includes a Bayesian popularity control.

```bash
# Tune on the 20,000-user v4 train/validation data. Test remains untouched.
.venv/bin/python -m src.experiments.item_cf_experiment

# Run only after freezing the validation winner. This evaluates the held-out
# test split without performing any additional parameter search.
.venv/bin/python -m src.experiments.final_item_cf_test
```

The experiment first performs a seeded random search, then refines the random
winner with a small local grid over neighboring K, minimum-support, and
shrinkage values. The search chooses the highest validation NDCG@10, with Recall@10 and catalog
coverage as secondary criteria. RMSE is reported separately because a model can
predict ratings accurately without placing the right unseen games near the top.
The checked-in v4 experiment evaluates all 20,000 selected users.

Concepts beyond the basic lecture formula are commented directly in the code:

- **Adjusted cosine** removes each user's average rating before comparing games.
- **Shrinkage** discounts similarities supported by very few common raters.
- **Baseline-residual scoring** combines deviations from a user's usual rating
  rather than averaging raw ratings from strict and generous users together.
- **Train/validation separation** prevents selecting parameters using the final
  test set. The test split is intentionally not accepted by the search command.

### Reduced-dataset validation results

The 20,000-user experiment evaluated 8 balanced random-search configurations
and 6 local-grid refinements. These validation results were used to freeze the
final configuration before opening the test split.

| Model / best similarity | Rating error (RMSE) ↓ | Liked-game hit rate (Recall@10) ↑ | Ranking quality (NDCG@10) ↑ | Game diversity (Coverage) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Bayesian popularity | 1.3892 | **0.0673** | **0.0515** | 0.0300 |
| Cosine | 1.3278 | 0.0097 | 0.0074 | 0.9965 |
| Adjusted cosine | **1.2506** | 0.0115 | 0.0090 | **0.9975** |
| Pearson | 1.4859 | 0.0505 | 0.0421 | 0.9860 |
| Jaccard | 1.3184 | 0.0206 | 0.0155 | 0.5315 |

Pearson is the current ItemCF ranking winner with 20 neighbors, minimum overlap
3, no shrinkage, and baseline-residual scoring. Bayesian popularity still has
the best validation NDCG@10 overall, while adjusted cosine has the lowest RMSE.
This disagreement illustrates why rating error alone is not sufficient for
evaluating Top-10 recommendation quality.

![Best validation ranking metrics](reports/v4/item_cf/ranking_metrics.png)

![Hyperparameter effects](reports/v4/item_cf/hyperparameter_effects.png)

### Final held-out test result

The frozen Pearson winner was then evaluated once on the 214,663 held-out test
interactions. No parameter was selected or changed using the test result.

| Split | Rating error (RMSE) ↓ | Liked-game hit rate (Recall@10) ↑ | Ranking quality (NDCG@10) ↑ | Game diversity (Coverage) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1.4859 | 0.0505 | 0.0421 | 0.9860 |
| **Test (final)** | **1.4797** | **0.0499** | **0.0413** | **0.9860** |

The test result remains close to validation: RMSE improved slightly, while
Recall@10 and NDCG@10 decreased slightly. The small gaps indicate that the
selected configuration generalizes stably to unseen interactions rather than
depending on a lucky validation split. The final report is stored in
`reports/v4/item_cf/final_test_summary.md`.

![Validation and final test comparison](reports/v4/item_cf/validation_vs_test.png)

## Matrix Factorization Experiment

The second personalized model represents every user and game with a learned
latent-factor embedding.  Global, user, and game biases capture broad rating
tendencies; the embedding dot product captures the remaining preference match.

```bash
# Tune only on train + validation. The test file is not accepted here.
.venv/bin/python -m src.experiments.matrix_factorization_experiment

# Evaluate the frozen validation winner once on held-out test data.
.venv/bin/python -m src.experiments.final_matrix_factorization_test
```

The deterministic random search compares latent factors, learning rate,
regularization, and epochs. The winner is selected by validation NDCG@10, with
Recall@10 and RMSE as tie breakers. The frozen 32-factor, learning-rate 0.02,
regularization 0.01, 18-epoch winner achieved validation/test NDCG@10 of
0.0201/0.0197 and RMSE of 1.2518/1.2512. Reproducible outputs are written to
`reports/v4/matrix_factorization/`.
