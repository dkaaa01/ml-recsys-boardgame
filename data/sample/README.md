# Small execution sample

This directory is a deterministic execution-check sample derived from the
current 20,000-user × 2,000-game experiment data. Selection uses train-only
statistics and seed 42, so validation and test information do not determine
which users or games enter the sample.

## Size

| Field | Count |
| --- | ---: |
| Users | 200 |
| Games | 200 |
| Train ratings | 7,140 |
| Validation ratings | 1,103 |
| Test ratings | 963 |
| Total ratings | 9,206 |
| Evaluation users | 200 |

Every retained user has at least five train ratings inside the sample catalog
and at least one 8+ rating in train, validation, and test.

## Files

- `train.csv`, `validation.csv`, `test.csv`: compact user-item-rating splits.
- `games.csv`: the 200 games and their current metadata/content fields.
- `users.csv`: compact IDs, source experiment IDs, and train activity.
- `eval_users.csv`: relevant-item counts for the common evaluation cohort.
- `popularity.csv`: Bayesian train-only ranking for the sample catalog.
- `subset_report.json`: selection rules, counts, strata, seed, and sparsity.

## Recreate

```bash
.venv/bin/python -m src.data.create_experiment_subset \
  --processed-dir data/experiment \
  --output-dir data/sample \
  --max-items 200 --max-users 200 \
  --min-user-train-ratings 5 \
  --relevant-threshold 8 --seed 42

.venv/bin/python -m src.models.popularity \
  --train data/sample/train.csv \
  --games data/sample/games.csv \
  --output-dir /tmp/boardgame-sample-baseline \
  --csv-output data/sample/popularity.csv
```

The sample is intended only for integration and execution checks. Final model
metrics must continue to use `data/experiment/`.
