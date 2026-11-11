# Board Game Recommendation System

This machine learning project builds personalized board game recommendations from BoardGameGeek (BGG) user ratings.

## Dataset Selection

This project uses the **2022 v3 snapshot** of `jvanelteren/boardgamegeek-reviews`.

- Approximately 19 million user-game ratings
- Approximately 410,000 users and 22,000 games
- Metadata including player count, playtime, complexity, categories, and mechanics
- Suitable for comparing collaborative filtering and content-based recommendation on the same dataset

The newer v4 release is approximately 5.4 GB, which is unnecessarily large for this project. The official BGG rank CSV only contains aggregated game-level scores and does not provide the user-game interactions required for personalized recommendations. The v3 snapshot provides a better balance between scale and project scope.

The original source is the [BoardGameGeek XML API](https://boardgamegeek.com/wiki/page/XML%20API), and the redistributed dataset is available through [Kaggle BoardGameGeek Reviews](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews). The BGG API terms should be reviewed before using the data.

## Preprocessing

The raw dataset contains **18,964,807 ratings across 21,839 games**, with interactions heavily concentrated among active users and popular games. The following rules keep enough interactions for model training while making the dataset manageable.

| Raw data | Criterion | Processing |
| --- | --- | --- |
| Missing user or game IDs and out-of-range ratings | Valid BGG ratings range from 1 to 10 | Removed 13 invalid rows |
| Some games have no detailed metadata or very few ratings | Ensure usable metadata and a minimum level of rating reliability | Kept games with metadata and at least 100 ratings |
| Ratings are concentrated among popular games | Balance training volume and computational cost | Selected the 3,000 most-rated eligible games |
| Many users have very little activity | Retain enough history for user preference learning | Removed users with fewer than 10 ratings in the selected catalog |
| Raw IDs are strings with large value ranges | Standardize model inputs | Encoded user and game IDs as zero-based integers |
| Random splits can change between runs | Ensure reproducibility | Split train/validation/test approximately 80/10/10 using a deterministic `(user, item)` hash |

The final dataset contains **212,439 users, 3,000 games, and 15,050,915 interactions**, with **97.64% sparsity**.

### Assignment experiment subset

The original processed dataset is preserved. A smaller experiment dataset is
selected using **training-split statistics only** to avoid validation/test
leakage:

- 1,500 games with the most training ratings
- users with at least 20 training ratings inside that catalog
- 10,000 users sampled proportionally from activity groups `20-39`, `40-79`,
  and `80+`
- compact zero-based IDs written separately under `data/experiment_10k/`

```bash
.venv/bin/python -m src.data.create_experiment_subset \
  --max-items 1500 --max-users 10000 \
  --min-user-train-ratings 20 --seed 42
```

The generated subset contains **899,544 interactions**: 721,930 train, 88,937
validation, and 88,677 test interactions. It has 10,000 users, 1,500 games,
and 94.00% matrix sparsity. Generated CSV files remain excluded from Git; the
selection code, report schema, and reproduction command are versioned.

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
# Fast, reproducible smoke experiment. It scans the full CSV files but retains
# only a seeded sample of users, then writes CSV/JSON results and PNG plots.
.venv/bin/python -m src.experiments.item_cf_experiment \
  --sample-users 2000 --trials 8 --grid-trials 6

# Main assignment experiment on the reduced dataset. Test remains untouched.
.venv/bin/python -m src.experiments.item_cf_experiment \
  --train data/experiment_10k/train.csv \
  --validation data/experiment_10k/validation.csv \
  --total-users 10000 --total-items 1500 --sample-users 10000 \
  --trials 16 --grid-trials 8 \
  --output-dir reports/item_cf_10k

# Run only after freezing the validation winner. This evaluates the held-out
# test split without performing any additional parameter search.
.venv/bin/python -m src.experiments.final_item_cf_test
```

The experiment first performs a seeded random search, then refines the random
winner with a small local grid over neighboring K, minimum-support, and
shrinkage values. The search chooses the highest validation NDCG@10, with Recall@10 and catalog
coverage as secondary criteria. RMSE is reported separately because a model can
predict ratings accurately without placing the right unseen games near the top.
The current CLI is intentionally capped at 10,000 sampled users. A full-data
multi-trial run will be enabled only after item-pair statistics are cached once
per similarity family instead of recomputed for every trial.

Concepts beyond the basic lecture formula are commented directly in the code:

- **Adjusted cosine** removes each user's average rating before comparing games.
- **Shrinkage** discounts similarities supported by very few common raters.
- **Baseline-residual scoring** combines deviations from a user's usual rating
  rather than averaging raw ratings from strict and generous users together.
- **Train/validation separation** prevents selecting parameters using the final
  test set. The test split is intentionally not accepted by the search command.

### Reduced-dataset validation results

The 10,000-user experiment evaluated 16 balanced random-search configurations
and 8 local-grid refinements. These validation results were used to freeze the
final configuration before opening the test split.

| Model / best similarity | Rating error (RMSE) ↓ | Liked-game hit rate (Recall@10) ↑ | Ranking quality (NDCG@10) ↑ | Game diversity (Coverage) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Bayesian popularity | 1.4138 | 0.0745 | 0.0534 | 0.0333 |
| Cosine | 1.3725 | 0.0162 | 0.0122 | 0.4413 |
| Adjusted cosine | 1.4001 | **0.0962** | **0.0711** | 0.1107 |
| Pearson | **1.3584** | 0.0267 | 0.0216 | **0.9973** |
| Jaccard | 1.3910 | 0.0660 | 0.0471 | 0.1447 |

Adjusted cosine is the current ranking winner with 120 neighbors, minimum
overlap 10, shrinkage 250, minimum similarity 0.1, and weighted-mean scoring.
Pearson has the lowest rating RMSE and widest coverage, illustrating why rating
error alone is not sufficient for evaluating Top-10 recommendation quality.

![Best validation ranking metrics](reports/item_cf_10k/ranking_metrics.png)

![Hyperparameter effects](reports/item_cf_10k/hyperparameter_effects.png)

### Final held-out test result

The adjusted-cosine winner was then evaluated once on the 88,677 held-out test
interactions. No parameter was selected or changed using the test result.

| Split | Rating error (RMSE) ↓ | Liked-game hit rate (Recall@10) ↑ | Ranking quality (NDCG@10) ↑ | Game diversity (Coverage) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1.4001 | 0.0962 | 0.0711 | 0.1107 |
| **Test (final)** | **1.3974** | **0.1005** | **0.0735** | **0.1107** |

The test result is close to, and slightly better than, the validation result.
This small gap indicates that the selected configuration generalizes stably to
unseen interactions rather than depending on a lucky validation split. The
final report is stored in `reports/item_cf_10k/final_test_summary.md`.

![Validation and final test comparison](reports/item_cf_10k/validation_vs_test.png)

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
Recall@10 and RMSE as tie breakers. Reproducible CSV, JSON, and plot outputs are
written to `reports/matrix_factorization_10k/`; the checked-in Markdown files
record the interpretation of the completed assignment run.
