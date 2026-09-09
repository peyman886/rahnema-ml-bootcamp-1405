"""Generate the dataset used in the week-3 sessions.

Same world as the week-2 dataset -- quick-commerce orders from four dark
stores -- but bigger, and this time with a target worth predicting:

    how many minutes will this delivery take?

It is synthetic on purpose. That lets me put specific, known traps in it and
then show exactly how much each one costs. Every trap below is one that has
actually cost somebody a quarter of work somewhere:

  * three columns that only exist *after* the delivery finished
    (`rating`, `tip_toman`, `actual_route_km`) -- target leakage
  * couriers who are consistently fast or slow, and who appear many times
    -- group leakage, and the reason naive target encoding overfits
  * a real improvement trend over the year -- random CV flatters a model
    that would be deployed on the future
  * one high-cardinality key (`courier_id`) with a long tail of couriers who
    barely appear -- unseen categories at predict time

Run:  python scripts/make_delivery_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 3
N_ORDERS = 12_000
N_COURIERS = 200
N_CUSTOMERS = 3_000
START = pd.Timestamp("2025-09-01")
DAYS = 300

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# store -> (city, region, extra minutes of drive time from this store)
STORES = {
    "Saadatabad": ("Tehran", "north", 0.0),
    "Punak": ("Tehran", "west", 1.5),
    "Gohardasht": ("Karaj", "west", 4.5),
    "Vakilabad": ("Mashhad", "east", 2.5),
}


def make_couriers(rng):
    """Each courier has a persistent speed of their own.

    This is what makes group leakage bite: if the same courier appears in
    both the training and the validation fold, the model can memorise that
    courier instead of learning anything about deliveries.
    """
    ids = [f"C{i:04d}" for i in range(1, N_COURIERS + 1)]
    return pd.DataFrame({
        "courier_id": ids,
        # Minutes faster (negative) or slower (positive) than an average courier.
        "courier_effect": rng.normal(0, 3.2, size=N_COURIERS),
        "courier_type": rng.choice(["motorcycle", "bicycle"], N_COURIERS, p=[0.78, 0.22]),
    })


def main():
    rng = np.random.default_rng(SEED)
    couriers = make_couriers(rng)

    # --- when ------------------------------------------------------------
    day = rng.integers(0, DAYS, size=N_ORDERS)
    # Two demand peaks: lunch and a bigger evening one.
    evening = rng.random(N_ORDERS) < 0.62
    hour = np.where(evening, rng.normal(20.0, 1.3, N_ORDERS),
                    rng.normal(13.0, 1.1, N_ORDERS)).clip(9, 23.5)
    order_time = (START + pd.to_timedelta(day, unit="D")
                  + pd.to_timedelta(hour * 60, unit="m")).round("min")

    # --- who and where ---------------------------------------------------
    store = rng.choice(list(STORES), size=N_ORDERS, p=[0.34, 0.26, 0.16, 0.24])
    # A busy core and a long tail of occasional riders. Some couriers will
    # appear only a handful of times, which is what makes unseen categories
    # and unstable per-courier statistics realistic.
    weights = rng.lognormal(0, 0.9, size=N_COURIERS)
    picked = rng.choice(N_COURIERS, size=N_ORDERS, p=weights / weights.sum())
    courier_id = couriers["courier_id"].to_numpy()[picked]
    courier_effect = couriers["courier_effect"].to_numpy()[picked]
    courier_type = couriers["courier_type"].to_numpy()[picked]
    # Customers have NO effect on delivery time here -- deliberately. Roughly
    # four orders each means naive target encoding of this column leaks a
    # quarter of each row's own answer back into its feature, and a column of
    # pure noise ends up looking like the best predictor in the model.
    customer_id = rng.choice([f"U{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
                             size=N_ORDERS)

    # --- the basket ------------------------------------------------------
    basket_size = 1 + rng.poisson(3.2, size=N_ORDERS)
    price_per_item = rng.lognormal(np.log(120_000), 0.45, size=N_ORDERS)
    basket_value = (basket_size * price_per_item).round(-3).astype("int64")

    # Straight-line distance, known the moment the order is placed. This is
    # the honest feature: it is an *estimate* of how far the courier will ride.
    distance_km = np.abs(rng.normal(2.4, 1.2, size=N_ORDERS)).clip(0.3, 9.0)
    # How much further they actually rode: one-way streets, closed roads, a
    # wrong turn. Only known once the trip is over.
    detour = rng.lognormal(np.log(1.30), 0.30, size=N_ORDERS).clip(1.0, 3.0)
    actual_route_km = (distance_km * detour).round(2)

    # What the store promises for picking, also known up front.
    prep_minutes = (2.0 + 0.85 * basket_size + rng.exponential(1.2, N_ORDERS)).round(1)

    weather = rng.choice(["clear", "rain", "snow"], N_ORDERS, p=[0.80, 0.17, 0.03])

    # --- the target ------------------------------------------------------
    store_extra = np.array([STORES[s][2] for s in store])
    # Deliveries get steadily faster as the fleet grows, and then drop again
    # when a new dispatch algorithm ships partway through. Together these are
    # what make random cross-validation flattering: shuffling the rows lets
    # the model see the future while it is being scored.
    trend = -5.5 * (day / DAYS) - 2.5 * (day >= 180)

    # Note what drives the target: the distance actually ridden, not the
    # estimate. That is exactly why `actual_route_km` is such a strong leak --
    # it is half the answer, recorded after the fact.
    delivery = (6.0
                + 3.4 * actual_route_km
                + store_extra
                + courier_effect
                + np.where(courier_type == "bicycle", 3.0, 0.0)
                + np.where(evening, 2.2, 0.0)
                + np.where(weather == "rain", 2.0, 0.0)
                + np.where(weather == "snow", 6.0, 0.0)
                + 0.25 * basket_size
                + trend
                + rng.exponential(1.6, size=N_ORDERS))
    # A few percent go badly wrong: wrong address, breakdown, lost courier.
    incident = rng.random(N_ORDERS) < 0.025
    delivery = delivery + incident * rng.uniform(8, 25, N_ORDERS)
    # A courier still has to reach the door and hand the bag over, so there is
    # a floor no combination of fast rider and short trip can go below.
    delivery = np.maximum(delivery, 3.0).round(1)

    # --- columns that only exist after the delivery ends -----------------
    # These are in the file because they are in the real table too; nobody
    # removes them for you. `actual_route_km` was computed above, because the
    # target depends on it.
    rating = np.clip(np.round(5.6 - 0.085 * delivery
                              + rng.normal(0, 0.5, N_ORDERS)), 1, 5)
    # Deliberately independent of the delivery time: this one leaks nothing
    # useful, and is here to show that "does it improve the score" is not the
    # test for whether a column is allowed.
    tip = np.where(rng.random(N_ORDERS) < 0.25,
                   (rng.lognormal(np.log(30_000), 0.5, N_ORDERS) / 1000).round() * 1000,
                   0).astype("int64")

    df = pd.DataFrame({
        "order_id": np.arange(500_001, 500_001 + N_ORDERS),
        "order_time": order_time,
        "store": store,
        "customer_id": customer_id,
        "courier_id": courier_id,
        "courier_type": courier_type,
        "weather": weather,
        "basket_size": basket_size,
        "basket_value_toman": basket_value,
        "distance_km": distance_km.round(2),
        "prep_minutes": prep_minutes,
        # --- after the fact, do not train on these ---
        "actual_route_km": actual_route_km,
        "rating": rating,
        "tip_toman": tip,
        # --- target ---
        "delivery_minutes": delivery,
    }).sort_values("order_time", ignore_index=True)

    # A little honest mess, so cleaning is not a theoretical exercise:
    # weather is sometimes not recorded, and a handful of prep times are lost.
    df.loc[rng.choice(df.index, 900, replace=False), "weather"] = np.nan
    df.loc[rng.choice(df.index, 240, replace=False), "prep_minutes"] = np.nan

    stores = pd.DataFrame(
        [{"store": k, "city": v[0], "region": v[1]} for k, v in STORES.items()])

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "deliveries.csv", index=False)
    stores.to_csv(DATA_DIR / "delivery_stores.csv", index=False)
    print(f"deliveries: {df.shape} -> {DATA_DIR / 'deliveries.csv'}")
    print(f"stores    : {stores.shape} -> {DATA_DIR / 'delivery_stores.csv'}")


if __name__ == "__main__":
    main()
