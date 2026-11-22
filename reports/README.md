# Result files

The PDF is submitted separately with the code folder. This directory contains the original figures, tables and saved model settings.

- comparison/ contains the final five-model test results.
- eda/ contains plots for the processed 248,457-user, 3,000-game dataset.
- baseline/ contains training-based Bayesian popularity scores.
- item_cf/, matrix_factorization/ and content_based/ contain validation searches and test results.
- hybrid/ contains the validation-weight results and English console-output figures.

The final test uses 20,000 users, 2,000 games and 10 recommendations per user. A test rating of at least 8 is relevant. Training-rated games are excluded. Player/time filters are used in the terminal example.

The catalog and activity groups are based on training counts. Evaluation eligibility also requires a positive rating in each split. The model settings are selected using validation scores. The test results describe one fixed split.

Source functions explain their inputs and outputs. The library and parameter descriptions are in [code explanations](../docs/code_explanations.md).
