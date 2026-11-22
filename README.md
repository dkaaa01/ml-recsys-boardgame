# Board Game Recommendation System

Machine Learning Term Project, Team 2

Kim Seongwon, Lee Minseo, Yoon Soim and Han Haram

## Run the program

Use Python 3.11 or later. Install the packages from requirements.txt in a virtual environment.

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

On Windows, use .venv/Scripts/python.exe in place of .venv/bin/python.

To try the recommender, run:

    .venv/bin/python recommend_cli.py --data-dir data/sample --scores outputs/scores.csv

Press Enter for a guest recommendation, or enter 0 to use an existing sample profile. Then enter the player count and maximum playtime in minutes. The sample has 200 users and 200 games. Guest ranking uses the number of training ratings. Existing profiles use saved hybrid scores and exclude games they rated in training.

## Recalculate the final results

    .venv/bin/python -m src.experiments.final_model_comparison

The smaller submission copy includes the sample data only. In that copy, first run `python prepare_data.py` to download the full experiment CSVs and saved rankings from the fixed result version.

This reads the saved recommendations and experiment CSVs. It checks that all five methods use the same 20,000 users, 2,000 games and seen-game exclusions, then writes the comparison CSV and figure under reports/comparison/.

To redraw the report charts:

    .venv/bin/python -m src.experiments.output_figures

| Model | Recall@10 | NDCG@10 | Coverage |
| --- | ---: | ---: | ---: |
| Bayesian popularity | 0.0692 | 0.0530 | 3.0% |
| Hybrid | 0.0554 | 0.0463 | 89.3% |
| ItemCF | 0.0499 | 0.0413 | 98.6% |
| Content-based | 0.0438 | 0.0352 | 81.6% |
| Matrix factorization | 0.0239 | 0.0197 | 80.6% |

A test rating of at least 8 counts as a liked game. Each method recommends 10 games. Player and time limits are used in the terminal program, while the final comparison uses the full unseen catalog.

## Code and data files

- src/data/ contains downloading, preprocessing, sampling, loading and plotting code.
- src/models/ contains popularity, ItemCF, matrix factorization and content-based models.
- src/experiments/ contains validation searches and final evaluations.
- src/metrics.py calculates the evaluation metrics.
- hybrid.py and hybrid_pipeline.py combine ItemCF and content scores.
- make_top10.py fits the saved settings and exports recommendations.
- recommend_cli.py runs the terminal program.
- data/experiment/ contains the 20,000-user, 2,000-game experiment inputs.
- data/sample/ contains the smaller terminal sample.
- outputs/top10_test.csv contains the four personalized models' final recommendations.
- outputs/scores.csv contains hybrid scores for the small terminal sample.
- reports/ contains raw figures, search tables and saved settings.

The submitted output PDF is provided separately with the code folder. The repository also has a copy under reports/Team2_Outputs.pdf.

[Function, library and parameter explanations](docs/code_explanations.md) are included with the source code. Functions also describe their inputs and outputs beside the implementation.

## Data preparation

We used version 4 of the Kaggle BoardGameGeek Reviews dataset, containing the 2025 files. The recorded raw file has 26,200,012 ratings. Cleaning and catalog selection produced 248,457 users, 3,000 games and 19,745,913 ratings.

The experiment subset contains 20,000 users, 2,000 games and 2,029,222 ratings: 1,599,055 training, 215,504 validation and 214,663 test ratings.

Catalog selection and user activity groups use training counts. Evaluation eligibility also requires a rating of at least 8 in each split. The split preserves positive ratings and is not chronological. Model settings are selected on validation results.

To rebuild the data from the original archive:

    .venv/bin/python -m src.data.download --dataset-version 4
    unzip -j data/raw/boardgamegeek-reviews.zip bgg-26m-reviews.csv games_detailed_info2025.csv -d data/raw
    .venv/bin/python -m src.data.preprocess
    .venv/bin/python -m src.data.create_experiment_subset --processed-dir data/processed --output-dir data/experiment --max-items 2000 --max-users 20000 --min-user-train-ratings 20 --seed 42
    .venv/bin/python -m src.models.popularity

The original archive and large intermediate data can be rebuilt. The experiment inputs needed for the final comparison are already included.

## Model experiments

The selected settings are saved in the model folders under reports/.

| Model | Selected settings |
| --- | --- |
| ItemCF | Pearson, 20 neighbors, minimum overlap 3, shrinkage 0, baseline-residual scoring |
| Matrix factorization | 32 factors, learning rate 0.02, regularization 0.01, 18 epochs, batch size 4,096 |
| Content-based | 4,000 description features, minimum document frequency 2, description weight 0.5, liked threshold 8 |
| Hybrid | 25% ItemCF and 75% content scores after per-user min-max scaling |

The recorded hybrid validation NDCG@10 values were 0.0464, 0.0435 and 0.0379 for ItemCF weights 0.25, 0.50 and 0.75. Weight 0.25 was selected.

To run the searches and fixed-setting evaluations:

    .venv/bin/python -m src.experiments.item_cf_experiment
    .venv/bin/python -m src.experiments.final_item_cf_test
    .venv/bin/python -m src.experiments.matrix_factorization_experiment
    .venv/bin/python -m src.experiments.final_matrix_factorization_test
    .venv/bin/python -m src.experiments.content_based_experiment --mode tune
    .venv/bin/python -m src.experiments.content_based_experiment --mode final

To regenerate personalized recommendations:

    .venv/bin/python make_top10.py --data-dir data/experiment --out-dir outputs/experiment

The full hybrid score export is much larger than the sample score file. A separate output directory keeps the sample scores available for the terminal example.

## Tests

    .venv/bin/python -m unittest discover -s tests -v

## References

- [BoardGameGeek Reviews, Kaggle](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews), version 4.
- [BoardGameGeek XML API](https://boardgamegeek.com/wiki/page/XML%20API), the original data source.
- [Weighted hybridization, ApX](https://apxml.com/courses/building-ml-recommendation-system/chapter-6-constructing-a-hybrid-recommendation-system/weighted-hybridization), the reference used for combining scores. Its alpha weights content, while our w weights ItemCF.
- Course material by Won Kim and Woong-Kee Loh, Machine-Learning-Recommendation.pdf, and the text/vector-space lectures.
- Official Python, NumPy, SciPy and Matplotlib documentation is linked in [code explanations](docs/code_explanations.md).

Team repository: https://github.com/dkaaa01/ml-recsys-boardgame
