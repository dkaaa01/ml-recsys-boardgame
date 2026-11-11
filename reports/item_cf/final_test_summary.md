# Final ItemCF test result

The validation winner was frozen before this evaluation. No test metric was
used to select or change the model or its hyperparameters.

## Frozen configuration

- Similarity: `pearson`
- Scoring: `baseline_residual`
- Neighbors: `20`
- Minimum overlap: `3`
- Shrinkage: `0.0`
- Minimum similarity: `0.0`

## Validation and final test metrics

| Split | Rating error (RMSE) ↓ | Liked-game hit rate (Recall@10) ↑ | Ranking quality (NDCG@10) ↑ | Game diversity (Coverage) ↑ |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1.4859 | 0.0505 | 0.0421 | 0.9860 |
| **Test (final)** | **1.4797** | **0.0499** | **0.0413** | **0.9860** |

- Test rating predictions: 214,663
- Test users with at least one relevant game: 20,000
- Fit time: 1.44 seconds
- Test evaluation time: 6.81 seconds
