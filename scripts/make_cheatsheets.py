"""Render the reference figures to assets/ so they can be printed or shared.

The notebooks draw the same figures inline; this script only exists so there
is a PNG to paste into Slack or stick above a monitor.

Run:  python scripts/make_cheatsheets.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bootcamp import (numpy_cheatsheet, pandas_cheatsheet, pipeline_diagram,
                      pipeline_anatomy, cv_schemes, sklearn_api,
                      preprocessing_sheet, feature_families, metrics_sheet,
                      leakage_gallery, scaling_matters, demand_problem,
                      case_leak_stories)

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    for name, fn in [("numpy_cheatsheet", numpy_cheatsheet),
                     ("pandas_cheatsheet", pandas_cheatsheet),
                     ("ds_pipeline", pipeline_diagram),
                     ("pipeline_anatomy", pipeline_anatomy),
                     ("cv_schemes", cv_schemes),
                     ("preprocessing_sheet", preprocessing_sheet),
                     ("feature_families", feature_families),
                     ("metrics_sheet", metrics_sheet),
                     ("sklearn_api", sklearn_api),
                     ("leakage_gallery", leakage_gallery),
                     ("scaling_matters", scaling_matters),
                     ("demand_problem", demand_problem),
                     ("case_leak_stories", case_leak_stories)]:
        out = ASSETS / f"{name}.png"
        fn(save_to=out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
