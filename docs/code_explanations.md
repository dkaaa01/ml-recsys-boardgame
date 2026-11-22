# Code explanations and references

This file explains how the project files work together and how the main library functions are used. Function descriptions in the code give the inputs, outputs and matrix shapes. The configuration classes explain each model parameter.

## Software architecture

1. `src/data/download.py` retrieves the pinned Kaggle archive and checks its digest. `preprocess.py` validates ratings, joins game metadata, assigns compact IDs and writes common splits. SQLite stores intermediate ratings on disk so the raw file can exceed memory.
2. `src/data/create_experiment_subset.py` selects the experiment catalog and activity groups using training counts. Evaluation eligibility also requires positive ratings in the held-out splits. `loader.py` loads sparse matrices. `eda.py` produces descriptive plots.
3. `src/models/` implements the Bayesian popularity baseline, item-item collaborative filtering, matrix factorization and content-based recommendation. `hybrid.py` combines normalized collaborative and content scores. `hybrid_pipeline.py` adapts the fitted models to a shared candidate array.
4. `src/experiments/` tunes configurations on validation data, evaluates frozen settings and writes tables and plots. `make_top10.py` exports the four personalized rankings. `final_model_comparison.py` adds the Bayesian baseline and checks the common evaluation conditions before scoring all five models. `output_figures.py` draws report figures from saved results and captured console text.
5. `recommend_cli.py` reads a profile, player count and time limit. It filters invalid or unsuitable games and ranks the remaining candidates. Guest ranking uses training rating counts. Profile ranking uses saved hybrid scores and excludes training-rated games.

The demo and offline comparison have distinct scopes. The demo uses 200 users and 200 games. The final comparison uses 20,000 users and 2,000 games without player/time filters. The guest demo's count ranking differs from the Bayesian baseline's adjusted-average ranking.

## Data and evaluation decisions

- Ratings must be finite and within 1 to 10. Exact duplicates are collapsed. A user-game pair with conflicting ratings is rejected. Missing metadata prevents a game from entering the catalog.
- Stable hashing assigns provisional splits. The per-user repair preserves liked games in the splits when sufficient positives exist. This is a label-aware evaluation design, rather than a chronological forecast.
- Game popularity and user activity are calculated from training ratings. The ranking cohort is additionally conditioned on at least one rating of 8 or above in each split. This uses held-out labels to define who can be evaluated. Model fitting uses training ratings, and configuration selection uses validation scores.
- Activity sampling preserves group proportions using largest-remainder allocation. The historical label `20-39` means the below-40 group. In the small execution sample, that group can include users with 5 to 19 training ratings.
- `Recall@10` is the mean of each user's matched test positives divided by that user's number of test positives. `NDCG@10` discounts hits at rank r by `1/log2(r+1)` and divides by the ideal top-10 discounted gain. Ratings of at least 8 are binary positives.
- `Coverage` counts distinct recommended games across all evaluated users and divides by the catalog size. It measures catalog reach. A high value does not establish that every user's list is diverse or relevant.
- Missing test ratings cannot confirm a recommendation as a hit. The saved metrics describe the observed held-out ratings. A single data split does not establish significance or future-user performance.

## Model implementation choices

### Bayesian popularity

The adjusted score is `(v * R + m * C) / (v + m)`. Here `v` is a game's training rating count, `R` its training mean, `C` the overall training mean and `m` the selected count quantile. This shrinks low-support means toward the global mean. It is an empirical shrinkage baseline, not a fitted posterior distribution. Recommendations apply each user's seen-game filter to the common ordering.

### Item-item collaborative filtering

Sparse products build item-pair support and similarities. The four variants are cosine, adjusted cosine, Pearson-style correlation on common raters and positive-set Jaccard. Minimum overlap discards weakly supported pairs. Shrinkage scales a similarity by `support / (support + shrinkage)`. Only the configured number of neighbors is retained.

Baseline-residual scoring starts from the user's training mean and adds a similarity-weighted sum of neighboring rating deviations from that mean. Absolute similarity weights form the denominator. The standalone model falls back to training popularity when no usable neighbors support a score. For the hybrid, a separate evidence mask prevents that fallback from being treated as collaborative evidence. The frozen experiment uses Pearson, 20 neighbors, minimum overlap 3 and shrinkage 0.

### Matrix factorization

The prediction is the global mean plus user bias, game bias and a user-game latent-vector dot product. Mini-batch updates minimize observed-rating error with regularization. Repeated user/game indices are accumulated with `numpy.add.at`, and inverse within-batch frequency controls repeated-index update magnitude. A seeded generator controls initialization and shuffling. Predictions are clipped to 1 to 10. Ranking reuses the model evaluator's tie handling to reproduce the saved rankings. The frozen configuration has 32 factors, learning rate 0.02, regularization 0.01, 18 epochs and batch size 4,096.

### Content-based recommendation

Descriptions are HTML-cleaned, lowercased and tokenized. Common game words are removed. Terms occurring in fewer than the configured number of documents are dropped. Description TF-IDF vectors and category/mechanic multi-hot vectors are separately normalized, scaled, concatenated and normalized again. Thus the implementation combines feature blocks, rather than averaging two final cosine scores.

The user profile is a weighted sum of training-liked game vectors. A rating r contributes weight `r - liked_threshold + 1`. The profile is normalized before cosine scoring. A user with no liked training games has a zero profile and tied zero raw cosine scores. For RMSE only, the predicted rating is `clip(user_mean + rating_scale * (cosine - catalog_average_similarity), 1, 10)`. The catalog-average similarity centers that rating conversion. The frozen configuration uses 4,000 description features, minimum document frequency 2, description weight 0.5 and a liked threshold of 8. `rating_scale=2.0` affects rating estimates, while ranking uses raw cosine scores.

### Weighted hybrid

For each user, scores are min-max scaled across the same eligible unseen catalog. Collaborative scores without neighbor evidence remain zero. The combined score is `w * ItemCF_normalized + (1 - w) * content_normalized`. Constant score ranges map to zero to avoid division by zero. Validation NDCG@10 selects w from 0.25, 0.50 and 0.75. The selected value is 0.25. Game ID resolves exact ties. The existing hybrid reference below uses alpha for the content contribution, so its weight direction differs from w here.

## Numerical libraries and their parameters

The runtime dependencies are NumPy, SciPy and Matplotlib. Their pinned versions are in `requirements.txt`. Model algorithms and metrics are implemented in the project source. Official API references explain the dependency operations listed below.

| API | Parameters used and purpose in this project |
| --- | --- |
| `numpy.asarray`, `ndarray.astype` | `a`, `dtype` and `copy` control array conversion, numeric precision and whether storage is reused. |
| `zeros`, `ones`, `empty`, `full` and corresponding `_like` functions | `shape` or an example array defines storage dimensions. `fill_value` and `dtype` create initialization vectors, masks and fallback scores. |
| `arange` | A stop value creates consecutive compact user/game indices. |
| `frombuffer` | `buffer` and `dtype` view typed rating-loader buffers without building Python-object arrays. |
| `bincount` | Integer indices, optional `weights` and `minlength` aggregate counts or totals for every user/game. |
| `diff`, `repeat` | `diff` obtains CSR row lengths. `repeat(a, repeats)` expands per-row information to each stored value. |
| `divide` | `out` and `where` compute supported means or normalized values while protecting empty denominators. |
| `sum`, `mean`, `max`, `min` | `axis` determines whether reduction is per user, game, feature or whole array. |
| `sqrt`, `abs`, `maximum` | Elementwise roots, absolute weights and denominator bounds support similarity normalization and RMSE. |
| `where`, `isfinite`, `any`, `flatnonzero` | Conditions and masks locate valid scores, liked ratings and evidence-supported candidates. |
| `isin` | `element` and `test_elements` identify seen games to exclude. |
| `clip` | Lower/upper rating bounds restrict explicit-rating predictions to 1 to 10. |
| `argpartition` | `a` and `kth` identify a top-score subset without sorting the entire catalog. |
| `lexsort` | Ordered key arrays resolve score ties deterministically. The last key is the primary sort key. |
| `sort`, `round`, `all`, `allclose`, `numpy.testing.assert_allclose` | Sorting produces stable ID lists, rounding formats output, and boolean/tolerance comparisons verify examples and tests. `assert_allclose(actual, desired, rtol, atol)` raises an assertion when values exceed the requested tolerance. `numpy.array` creates explicit test arrays. |
| `diag`, `fill_diagonal`, `outer` | Read/remove self-similarity or form pairwise normalization terms. `outer(a,b)` creates all pair products. |
| `einsum` | The subscript expression `ij,ij->i` computes one latent-vector dot product per observed pair. |
| `add.at` | `a`, `indices` and `b` specify the array, positions and values to add. Repeated positions receive all their gradient contributions. |
| `quantile` | `a` and `q` choose the training-count prior strength for popularity shrinkage. |
| `random.default_rng` | `seed` creates the reproducible random generator. `choice` samples configurations/users, `normal` initializes factors, `shuffle` permutes training order, and `integers` creates synthetic examples where used. |
| `ndarray.copy`, `ravel`, `tolist`, indexing, transpose and `@` | Copy isolates score modifications, ravel converts results to one dimension, tolist creates ordinary output lists, and matrix multiplication computes similarities and predictions. |
| `scipy.sparse.coo_matrix` | `(values, (rows, columns))`, `shape` and `dtype` construct a rating matrix from triplets. Missing entries stay unstored. |
| `scipy.sparse.csr_matrix` | Input data, `shape` and `dtype` create a row-oriented sparse matrix for users or feature vectors. |
| `tocsr`, `tocoo`, `astype`, `getrow` | Convert sparse layout/precision and retrieve a user's stored ratings. `getrow(i)` selects row i. |
| Sparse `data`, `indices`, `indptr`, `nnz`, `shape` | Stored values, column indices, row boundaries, stored-value count and dimensions define CSR access and matrix sparsity. Stored values can include explicit zeros until they are eliminated. |
| Sparse `multiply`, `sum`, `mean`, `maximum`, transpose and `@` | Elementwise multiplication and axis reductions build moments, while sparse matrix multiplication builds item similarities and user-content scores. |
| Sparse `toarray`, `todense`, `eliminate_zeros`, `sum_duplicates`, `sort_indices`, `getnnz` | Convert results for ranking, remove stored zeros, combine duplicate coordinates, sort each row's indices or count stored entries along an axis. |
| `scipy.sparse.diags`, `hstack` | `diagonals` creates row scaling operators. A sequence of matrices and `format='csr'` concatenates description and tag feature blocks. |
| `issparse`, `isspmatrix_csr` | Detect sparse data and validate row-oriented inputs before numeric operations. |
| `matplotlib.pyplot.subplots` | `nrows`, `ncols` and `figsize` define the plot grid and canvas dimensions. |
| Axes `bar`, `barh`, `hist`, `scatter` | Positions/values define bars or points. `bins` controls histogram intervals, `s` marker size, and `c` point color for parameter searches. |
| Axes labels, limits, ticks, scales, `legend`, `grid`, `text` and `annotate` | Titles and axis values identify metrics, trial IDs and configurations. `symlog` preserves zero while displaying larger shrinkage values. |
| Figure `tight_layout`, `subplots_adjust`, `suptitle`, `text`, `colorbar`, `savefig` | Spacing parameters avoid overlap. `dpi` controls export resolution, and the path controls the saved figure. |
| `pyplot.style.use`, `pyplot.close` | Select plot appearance and release the finished figure. |
| `matplotlib.use` | The backend name `Agg` selects file rendering for the output-figure exporter. It is set before importing pyplot. |
| `matplotlib.patches.Patch` | `facecolor` and `label` create similarity-family legend entries. |
| `Figure`, `FigureCanvasAgg` | `figsize` sets a figure and the Agg canvas renders PNG output without an interactive display. |

Plot method parameters used in the figure writers are specified below. Each method changes the current figure or axes; the resulting image is saved with `savefig`.

| Method | Parameters used |
| --- | --- |
| `Axes.bar` / `Axes.barh` | `x` and `height`, or `y` and `width`, supply positions and values. `width`/`height` controls bar thickness and `color` selects the series color. |
| `Axes.hist` / `Axes.scatter` | Histogram `x`, `bins` and `color` define counts. Scatter `x`, `y`, `s`, `c`, `cmap`, `alpha`, `edgecolor` and `linewidth` control data positions and configuration encodings. |
| `Axes.set`, `set_title`, `set_xlabel`, `set_ylabel` | `title`, `xlabel`, `ylabel` or the label argument identifies the quantity. Title `loc`, `fontsize`, `weight`, `color` and `pad` control placement and text formatting. |
| `Axes.set_xlim`, `set_xticks`, `set_yticks` | Bounds define the displayed interval. Tick positions and optional labels identify each model or trial. `fontsize` controls label size. |
| `Axes.set_xscale` | The scale name selects linear, log or symlog coordinates. `linthresh` preserves a linear interval around zero for shrinkage values. |
| `Axes.text` / `Axes.annotate` | `x`, `y`, `s` or `text`, `xy` supply label content and anchor. `xytext`, `textcoords`, `ha`, `va`, `fontsize` and `color` control offset, alignment and appearance. |
| `Axes.legend` / `Axes.grid` | `handles` and `title` identify series. Grid `axis` and `alpha` select the reference lines and transparency. |
| `Axes.invert_yaxis`, `tick_params`, spine `set_visible` | Inversion puts the first model at the top. `axis`, `length` and `pad` control tick marks. The visibility boolean removes unneeded borders. |
| Bar `get_x`, `get_width` | Return a rectangle's horizontal position and width so a label can be centered on its bar. |
| `Figure.subplots` / `subplots_adjust` | Row/column counts create panels. `left`, `right`, `top`, `bottom` and `wspace` set panel placement. |
| `Figure.text`, `suptitle`, `supxlabel`, `supylabel` | Text and figure-relative coordinates place titles, shared labels and captions. `fontsize`, `weight`, `color`, `family` and `linespacing` control formatting. |
| `Figure.colorbar` / `savefig` | The plotted object, target `ax` and `label` create a parameter scale. The output path, `dpi`, `bbox_inches` and `pad_inches` determine image resolution, bounds and padding. |

## Standard-library modules

| Module or API | Parameters and use |
| --- | --- |
| `argparse.ArgumentParser`, `add_argument`, `parse_args` | Define command-line flags, `type`, `default`, `choices`, `action` and help strings. Paths, seeds, thresholds, trial counts and output directories are parsed by the executable modules. |
| `pathlib.Path` | Represents input/output locations. `open(mode, encoding, newline)`, `mkdir(parents, exist_ok)`, `resolve`, `exists`, `replace`, `with_suffix` and path joining manage files predictably. |
| `csv.DictReader`, `DictWriter`, `writer` | A file handle, optional field names, `writerow` and `writerows` read/write the documented CSV columns. |
| `json.load`, `loads`, `dump`, `dumps` | File or string input serializes configurations and result dictionaries. `indent` and `ensure_ascii` control readable output. |
| `dataclasses.dataclass`, `asdict`, `fields` | `frozen=True` protects configurations. `asdict` saves named settings, and `fields(cls)` reads their types when loading JSON. |
| `collections.Counter`, `defaultdict` | Counter aggregates ratings/tags. `defaultdict(factory)` creates empty per-user sets, dictionaries or lists on first use. |
| `sqlite3.connect`, connection `execute`/`executemany`, `commit`, `close` | A database path and SQL parameters store ratings, group duplicate pairs and stream selected observations without retaining the raw CSV in memory. Bound values keep data separate from SQL syntax. |
| `hashlib.blake2b` | `payload` and `digest_size=8` produce stable split buckets; `digest()` supplies bytes converted to an integer. |
| `hashlib.md5`, `sha256`, digest `update`/`hexdigest` | Chunked byte inputs check archive identity or record source-file provenance. MD5 here checks file identity. |
| `urllib.request.Request`, `urlopen` | URL and request headers fetch the pinned archive. The response is read as bytes. |
| `shutil.copyfileobj`, `rmtree` | File handles and `length` stream downloads; a path and `ignore_errors` clean a temporary preprocessing directory. |
| `array.array` | Type codes select compact integer/float buffers for rating triplets before sparse conversion. |
| `re.compile` and pattern `findall`/`sub` | A pattern selects description tokens or removes HTML markup. `html.unescape` decodes character entities. |
| `ast.literal_eval` | Converts the metadata's string representation of a list into Python tag values. |
| `math.isfinite`, `sqrt`, `log`, `log2` | Check ratings, calculate errors, weight text terms and discount ranking positions. |
| `decimal.Decimal`, `quantize` | Exact string-based decimal input and `ROUND_HALF_UP` display coverage percentages consistently. |
| `itertools.islice` | An iterable and stop count take the first K eligible popular games. |
| `time.perf_counter` | Measures elapsed fitting/evaluation time. Timing varies by machine and is descriptive. |
| `os.environ.setdefault` | A variable name and default path direct plotting caches to a writable project directory. |
| `random.Random` | A seed creates repeatable tiny synthetic data for the optional CLI format example. These records are separate from the BGG experiment. |
| `typing`, `collections.abc`, `__future__.annotations` | Type annotations describe accepted data and return values, including callables and sparse/mapping interfaces. |
| `unittest.TestCase`, `unittest.mock.patch`, `tempfile.TemporaryDirectory` | The tests use assertions, scoped dependency replacement and temporary output folders. `patch` targets a named function with a controlled replacement. The suite runs with Python's standard-library unittest runner. |

## References

- **Course material.** Won Kim and Woong-Kee Loh, *Machine-Learning-Recommendation.pdf*: similarity metrics, item-based filtering, matrix factorization, content-based recommendation and hybrids. The text/vector-space lecture materials provide the TF-IDF and cosine background. Course PDFs remain course-provided material.
- **Dataset.** Jeroen van Elteren, [BoardGameGeek Reviews on Kaggle](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews), dataset version 4, 2025 files. [BoardGameGeek XML API](https://boardgamegeek.com/wiki/page/XML%20API) describes the original source. The download script records the archive digest.
- **Hybrid reference.** ApX, [Weighted Hybridization in Recommenders](https://apxml.com/courses/building-ml-recommendation-system/chapter-6-constructing-a-hybrid-recommendation-system/weighted-hybridization). Background for combining two recommendation scores with a weight.
- **Python APIs.** [Python standard-library reference](https://docs.python.org/3/library/) covers file I/O, CSV, JSON, SQLite, hashing, command-line arguments and unittest.
- **NumPy APIs.** [NumPy reference](https://numpy.org/doc/stable/reference/) and [`numpy.add.at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html) document arrays, random generators and repeated-index accumulation.
- **Sparse matrices.** [SciPy sparse-matrix reference](https://docs.scipy.org/doc/scipy/reference/sparse.html) documents the sparse representations and operations used in training.
- **Plotting.** [Matplotlib API reference](https://matplotlib.org/stable/api/index.html) documents figure, axes and image-export operations.

The experiment values are stored in the CSV and JSON files under `reports/`.
