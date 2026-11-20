# Experiment outputs

[Team2_Outputs.pdf](Team2_Outputs.pdf) collects the experiment summary, model comparison, descriptive plots, model-search results and terminal demonstration outputs.

## Common final comparison

The experiment contains 20,000 users, 2,000 games and 2,029,222 ratings. Catalog/activity selection uses training counts. Evaluation eligibility additionally requires a positive rating in each split. Ratings of at least 8 count as relevant. Every model recommends 10 games after excluding each user's training-rated games.

| Model | Recall@10 | NDCG@10 | Catalog coverage |
| --- | ---: | ---: | ---: |
| Bayesian popularity | 0.0692 | 0.0530 | 3.0% |
| Hybrid | 0.0554 | 0.0463 | 89.3% |
| ItemCF | 0.0499 | 0.0413 | 98.6% |
| Content-based | 0.0438 | 0.0352 | 81.6% |
| Matrix factorization | 0.0239 | 0.0197 | 80.6% |

The Bayesian baseline leads both ranking metrics on this split. Hybrid leads the four personalized models. ItemCF reaches the largest share of the catalog. Coverage is the union of games recommended across users.

The comparison CSV and chart are produced by the final_model_comparison module. It checks the common cohort, candidates, seen-item exclusion and saved rankings before calculating the metrics. Recalculation uses frozen rankings and does not tune on test results.

## Model-specific files

- eda/ contains exploratory plots for the 248,457-user, 3,000-game processed dataset.
- baseline/ contains the training-only Bayesian score ordering and prior settings.
- item_cf/, matrix_factorization/ and content_based/ contain validation searches, frozen settings, held-out metrics and figures. ItemCF trial plots include its Bayesian validation control.
- hybrid/ contains recorded demo screens. The CLI uses the separate 200-user, 200-game sample. Guest scores are training counts, while profiles use saved hybrid scores.

Player/time filters apply to the demo. The final five-model comparison evaluates the common unseen catalog without these filters. The recorded search and test results use one fixed split; they do not establish statistical significance or live-user satisfaction.

Source files contain parameter and return-value documentation. [Code explanations and references](../docs/code_explanations.md) explain the architecture and library APIs.
