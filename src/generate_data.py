from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "housing.csv"


def generate_dataset(n_samples: int = 1200, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    area = rng.integers(550, 4200, n_samples)
    bedrooms = np.clip(np.round(area / 650 + rng.normal(0, 0.8, n_samples)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n_samples) + rng.choice([0, 1], n_samples), 1, 5).astype(int)
    age = rng.integers(0, 35, n_samples)
    distance = np.round(rng.uniform(1, 35, n_samples), 1)
    location = rng.choice(["Central", "Suburban", "Outskirts"], n_samples, p=[0.25, 0.50, 0.25])
    parking = rng.choice(["Yes", "No"], n_samples, p=[0.70, 0.30])
    furnished = rng.choice(["Furnished", "Semi-Furnished", "Unfurnished"], n_samples, p=[0.30, 0.45, 0.25])

    location_bonus = {"Central": 2_000_000, "Suburban": 700_000, "Outskirts": 0}
    furnished_bonus = {"Furnished": 450_000, "Semi-Furnished": 200_000, "Unfurnished": 0}

    price = (
        area * 7_500
        + bedrooms * 350_000
        + bathrooms * 250_000
        - age * 55_000
        - distance * 28_000
        + np.array([location_bonus[x] for x in location])
        + np.array(furnished_bonus[x] for x in furnished)
        + np.where(parking == "Yes", 300_000, 0)
        + rng.normal(0, 450_000, n_samples)
    )
    price = np.maximum(price, 1_500_000).round(-3).astype(int)

    return pd.DataFrame({
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "distance_km": distance,
        "location": location,
        "parking": parking,
        "furnished": furnished,
        "price": price,
    })


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT}")
