# ItemCF reduced-dataset validation summary

This report uses all users in the reduced assignment dataset for training and validation. The test split was not read. 14 ItemCF configurations
were evaluated in addition to the Bayesian popularity baseline.

## Best ItemCF configuration

- Similarity: `pearson`
- Scoring: `baseline_residual`
- Neighbors: `20`
- Minimum overlap: `3`
- Shrinkage: `0.0`
- RMSE: 1.4859
- Recall@10: 0.0505
- NDCG@10: 0.0421
- Catalog coverage: 0.9860

The best ItemCF NDCG@10 is lower than the Bayesian popularity control in this
reduced validation dataset (0.0515). This is a useful diagnostic, not a
reason to inspect the test set. The winning configuration can now be frozen before one final test evaluation.

## Visual outputs

- `ranking_metrics.png`: Recall, NDCG, and coverage on separate axes.
- `rmse_vs_ndcg.png`: rating-prediction error versus ranking quality.
- `search_trials.png`: random-search and local-grid validation trials.
- `hyperparameter_effects.png`: shrinkage, neighbor K, and support effects by similarity.
