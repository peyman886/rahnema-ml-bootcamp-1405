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

## Week 3 — scikit-learn, validation and leakage

One continuous session with lunch in the middle: build a pipeline before the
break, then spend the afternoon proving its score was too good.

| # | Notebook | What it is |
|---|---|---|
| 04 | [`04_sklearn_pipeline_lab.ipynb`](notebooks/04_sklearn_pipeline_lab.ipynb) | end to end on a messy table: metric, split, baseline, `ColumnTransformer`, `Pipeline`, cross-validation |
| 05 | [`05_validation_and_leakage_workshop.ipynb`](notebooks/05_validation_and_leakage_workshop.ipynb) | four ways your score lies: future columns, group leakage, time leakage, target encoding — then feature engineering and error analysis |

Notebook 05 is the one to read if you only read one. Each leak is measured
rather than described, and the target-encoding section makes a column with
literally zero information look like the best feature in the model.

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
