# Rahnema College — ML Bootcamp 1405

Lab and workshop material for the sessions I teach in the bootcamp.
Everything here is meant to be read as much as run: the notebooks are
committed with their outputs, so you can follow the argument on GitHub before
you install anything.

## Week 2 — NumPy, pandas, matplotlib

| # | Notebook | What it is |
|---|---|---|
| 00 | [`00_check_setup.ipynb`](notebooks/00_check_setup.ipynb) | environment check — **run this before the session** |
| 01 | [`01_numpy_pandas_matplotlib_lab.ipynb`](notebooks/01_numpy_pandas_matplotlib_lab.ipynb) | the lab: NumPy, pandas, matplotlib and seaborn |
| 02 | [`02_data_science_pipeline.ipynb`](notebooks/02_data_science_pipeline.ipynb) | the nine stages of a data science project, with EDA in depth |

## Week 3 — scikit-learn, features, validation and metrics

One continuous session with lunch in the middle. Each half pairs a concepts
notebook with the same ideas applied to a real internal forecasting problem.

| # | Notebook | What it is |
|---|---|---|
| 04 | [`04_preprocessing_and_features.ipynb`](notebooks/04_preprocessing_and_features.ipynb) | scikit-learn's one interface, the preprocessing menu, `ColumnTransformer` and `Pipeline`, and the families of features worth building |
| 05 | *case study* | the same thing on a real demand-forecasting panel — 90% zeros, 1.8M rows, and two leaks found the hard way |
| 06 | [`06_modelling_and_evaluation.ipynb`](notebooks/06_modelling_and_evaluation.ipynb) | fitting a model, four ways your score lies, which cross-validation scheme, which metric, error analysis |
| 07 | *case study* | the real experiments — model ladder, time splits, WMAPE, overstock/understock, and the grain that reversed the ranking |

Notebook 06 is the one to read if you only read one. Each leak is measured
rather than described, and the target-encoding section makes a column with
literally zero information look like the best feature in the model.

The two case-study notebooks read an internal dataset that is not in this
repository, so they are shown in the session rather than published. Everything
they demonstrate is in 04 and 06 with data you can run.

Answers to the exercises are published here after the in-person session. Try
them first.

Week 4 moves to neural networks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Then open `notebooks/00_check_setup.ipynb` and run it. If anything there fails,
ask in the channel **before** the in-person day — we are not spending lab time
on `pip`.

No internet is needed during the sessions: the datasets live in `data/`.

The notebooks ship with their outputs so they render on GitHub. Run
**Restart & Run All** yourself anyway — reading someone else's outputs teaches
you very little.

## The cheat sheets

`assets/` holds the reference figures as PNGs, sized for printing:

| Figure | Covers |
|---|---|
| [`numpy_cheatsheet.png`](assets/numpy_cheatsheet.png) | shapes, axes, indexing, masks, broadcasting |
| [`pandas_cheatsheet.png`](assets/pandas_cheatsheet.png) | anatomy, selection, the first-look ritual, groupby, merge |
| [`ds_pipeline.png`](assets/ds_pipeline.png) | the nine stages of a project |
| [`pipeline_anatomy.png`](assets/pipeline_anatomy.png) | what `ColumnTransformer` + `Pipeline` do to your columns, and why it has to be one object |
| [`cv_schemes.png`](assets/cv_schemes.png) | KFold / Stratified / Group / TimeSeries, drawn fold by fold |
| [`preprocessing_sheet.png`](assets/preprocessing_sheet.png) | the menu: missing values, scaling, categoricals, skew, dates |
| [`feature_families.png`](assets/feature_families.png) | the families of features worth building on a panel |
| [`metrics_sheet.png`](assets/metrics_sheet.png) | regression metrics, what each hides, and the grain question |

They are drawn by code rather than in a design tool, so they stay in version
control and the numbers on them match the numbers the notebooks print. See
`bootcamp/cheatsheets.py`.

## Regenerating things

```bash
python scripts/make_data.py           # week 2 dataset
python scripts/make_delivery_data.py  # week 3 dataset
python scripts/make_cheatsheets.py    # assets/*.png
```

## Layout

```
notebooks/   the sessions, in order
bootcamp/    helper module -- only draws the reference figures
data/        small synthetic datasets, committed on purpose
assets/      the cheat sheets as PNG, for printing
scripts/     regenerate data/ and assets/
```

## A note on the datasets

Both are synthetic, generated with a fixed seed, and both have deliberate
defects.

`data/qcommerce_orders.csv` (week 2) has dates stored as strings, inconsistent
category labels, a duplicated batch of rows, one impossible delivery time, and
ratings that are missing in a way that correlates with how long the customer
waited.

`data/deliveries.csv` (week 3) is built around a prediction target and carries
four planted traps: three columns that only exist after the delivery finished,
couriers who repeat often enough to be memorised, a real improvement trend over
the period, and a high-cardinality customer key with no signal in it at all.
`scripts/make_delivery_data.py` documents each one — read it after the session,
not before.

## Licence

Teaching material, free to reuse and adapt. Attribution appreciated.
