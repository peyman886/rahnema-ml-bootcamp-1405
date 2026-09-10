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

| # | Notebook | What it is |
|---|---|---|
| 04 | [`04_from_table_to_model.ipynb`](notebooks/04_from_table_to_model.ipynb) | the whole middle of a project: scikit-learn's one interface, preprocessing, features, `Pipeline`, models, the four leaks, cross-validation, metrics, error analysis |
| 05 | *case study* | the same ideas on a real demand-forecasting problem — 1.8M rows, 90% zeros, two leaks that actually happened, and an evaluation grain that mattered more than the model |

Notebook 04 is built the same way as the NumPy and pandas labs: a **sheet** per
section, then small examples to run and break, plus 20 numbered exercises you
can work through on your own.

The case study reads an internal dataset that is not in this repository, so it
is shown in the session rather than published. Everything it demonstrates is in
notebook 04 with data you can run.

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
| [`sklearn_api.png`](assets/sklearn_api.png) | the estimator interface, and what lives behind it |
| [`preprocessing_sheet.png`](assets/preprocessing_sheet.png) | the menu for one column: missing values, scaling, categoricals, skew, dates |
| [`scaling_matters.png`](assets/scaling_matters.png) | which model families care about scale, and why |
| [`leakage_gallery.png`](assets/leakage_gallery.png) | the four ways a validation score lies |
| [`cv_schemes.png`](assets/cv_schemes.png) | which split, decided by three questions |
| [`metrics_sheet.png`](assets/metrics_sheet.png) | regression metrics, what each hides, and the grain question |
| [`feature_families.png`](assets/feature_families.png) | the families of features worth building on a panel |

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
