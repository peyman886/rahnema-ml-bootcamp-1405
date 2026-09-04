"""Generate the toy dataset used in the week-2 lab.

It is synthetic on purpose: small enough to read on a projector, and the
relationships in it are ones we can point at ("of course delivery is slower
from Gohardasht, look at the distance"). The three small defects near the
bottom of this file are deliberate -- they are what the first-look ritual in
the notebook is supposed to catch.

Run:  python scripts/make_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 12
N_ORDERS = 240
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# store -> (city, region, opened_year, area_m2, minutes of extra drive time)
STORES = {
    "Saadatabad": ("Tehran", "north", 2022, 310, 0.0),
    "Punak": ("Tehran", "west", 2023, 250, 2.5),
    "Gohardasht": ("Karaj", "west", 2024, 190, 6.0),
    "Vakilabad": ("Mashhad", "east", 2023, 275, 3.5),
}


def make_stores() -> pd.DataFrame:
    rows = [
        {"store": name, "city": city, "region": region,
         "opened_year": year, "area_m2": area}
        for name, (city, region, year, area, _) in STORES.items()
    ]
    return pd.DataFrame(rows)


def make_orders(rng: np.random.Generator) -> pd.DataFrame:
    names = list(STORES)
    # Saadatabad is the oldest and busiest store, Gohardasht the newest.
    store = rng.choice(names, size=N_ORDERS, p=[0.34, 0.28, 0.16, 0.22])

    start = pd.Timestamp("2025-08-02")
    day_offset = rng.integers(0, 21, size=N_ORDERS)
    # Q-commerce demand is bimodal: a lunch peak and a bigger evening peak.
    peak = rng.random(N_ORDERS) < 0.62
    hour = np.where(peak, rng.normal(20.0, 1.3, N_ORDERS), rng.normal(13.0, 1.1, N_ORDERS))
    hour = hour.clip(9, 23.5)
    order_time = (start + pd.to_timedelta(day_offset, unit="D")
                  + pd.to_timedelta(hour * 60, unit="m")).round("min")

    basket_size = 1 + rng.poisson(3.2, size=N_ORDERS)
    price_per_item = rng.lognormal(mean=np.log(120_000), sigma=0.45, size=N_ORDERS)
    basket_value = (basket_size * price_per_item).round(-3)

    picking = 2.0 + 0.9 * basket_size + rng.exponential(1.6, size=N_ORDERS)

    courier = rng.choice(["motorcycle", "bicycle"], size=N_ORDERS, p=[0.78, 0.22])
    extra_drive = np.array([STORES[s][4] for s in store])
    # Right-skewed by construction: most deliveries are fine, a few are awful.
    delivery = (7.0 + extra_drive
                + 0.35 * basket_size
                + np.where(courier == "bicycle", 3.2, 0.0)
                + np.where(peak, 2.4, 0.0)
                + rng.exponential(3.0, size=N_ORDERS))
    # A few percent go badly wrong -- wrong address, breakdown, lost courier.
    # This is what makes p99 interesting, and it is a real effect, not noise.
    incident = rng.random(N_ORDERS) < 0.045
    delivery = delivery + incident * rng.uniform(12, 38, size=N_ORDERS)

    # Ratings react to how long people waited.
    latent = 5.4 - 0.085 * delivery + rng.normal(0, 0.55, size=N_ORDERS)
    rating = np.clip(np.round(latent), 1, 5)
    # ... and whether someone bothers to rate at all also depends on the wait:
    # annoyed customers are the ones who leave feedback. So the ratings we keep
    # are NOT a random sample of orders -- which is the point of the missing
    # data section in notebook 02.
    p_missing = 1.0 / (1.0 + np.exp(-(1.75 - 0.135 * delivery)))
    rating = np.where(rng.random(N_ORDERS) < p_missing, np.nan, rating)

    df = pd.DataFrame({
        "order_id": np.arange(100_001, 100_001 + N_ORDERS),
        "order_time": order_time,
        "store": store,
        "courier_type": courier,
        "basket_size": basket_size,
        "basket_value_toman": basket_value.astype("int64"),
        "picking_minutes": picking.round(1),
        "delivery_minutes": delivery.round(1),
        "rating": rating,
    }).sort_values("order_time", ignore_index=True)

    # --- the three deliberate defects -------------------------------------
    # 1. inconsistent category labels, the way a real form would produce them
    messy_idx = rng.choice(df.index, size=26, replace=False)
    df.loc[messy_idx, "courier_type"] = (
        df.loc[messy_idx, "courier_type"].str.capitalize() + " "
    )
    # 2. a few duplicated rows, as if the ingestion job ran twice
    dupes = df.sample(4, random_state=SEED)
    df = pd.concat([df, dupes], ignore_index=True).sort_values(
        "order_time", ignore_index=True)
    # 3. one impossible delivery time, to make the describe() step pay off
    df.loc[df.index[7], "delivery_minutes"] = -1.0
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    orders = make_orders(rng)
    stores = make_stores()
    orders.to_csv(DATA_DIR / "qcommerce_orders.csv", index=False)
    stores.to_csv(DATA_DIR / "qcommerce_stores.csv", index=False)
    print(f"orders: {orders.shape} -> {DATA_DIR / 'qcommerce_orders.csv'}")
    print(f"stores: {stores.shape} -> {DATA_DIR / 'qcommerce_stores.csv'}")


if __name__ == "__main__":
    main()
