"""
Adaptive Revenue Engine (ARE)
Product : Wireless Bluetooth Earbuds
Data Generator

Creates a synthetic market data of the product
for causal inference and reinforment learning.
"""

import numpy as np
import pandas as pd

# DATA GENERATION
def gen_data(n = 5000):

    np.random.seed(42)

    # PRODUCT FEATURES

    base_price = np.random.uniform(
        150, 
        800, 
        n
    )
    
    season = np.random.choice(
        [0, 1, 2],
        n,
        p=[0.27, 0.40, 0.33]
    )

    noise_cancellation = np.random.choice(
        [0, 1],
        n
    )

    brand_strength = np.random.uniform(
        0,
        1,
        n
    )

    battery_life = np.random.uniform(
        4,
        40,
        n
    )

    # SEASON IMPACT

    season_multiplier = np.where(
        season == 0,
        0.85,
        np.where(season == 1, 1.0, 1.25)
    )

    # BATTERY PREMIUM

    battery_effect = (
        120 *
        np.log1p(battery_life) / np.log1p(40)
    )

    # FINAL PRICE

    price = (
        base_price * season_multiplier
        + battery_effect
        + 250 * (brand_strength ** 4)
        + noise_cancellation * 150
        + np.random.normal(0, 50, n)
    ).round(2)

    competitor_price = (
        price *
        np.random.uniform(
            0.85,
            1.15,
            n
        )
    ).round(2)

    inventory = np.random.randint(
        100,
        1500,
        n
    )

    df = pd.DataFrame({

        "season": season,

        "noise_cancellation": noise_cancellation,

        "brand_strength": brand_strength,

        "battery_life": battery_life,

        "price": price,

        "competitor_price": competitor_price,

        "inventory": inventory
    })

    # DEMAND GENERATION

    season_effect = np.where(
        df["season"] == 0,
        np.random.uniform(-420, -120, n),
        np.where(
            df["season"] == 1,
            np.random.uniform(-100, 100, n),
            np.random.uniform(120, 600, n)
        )
    )

    demand = (
        np.random.uniform(3000, 6000, n)
        + (-3.2 * df["price"])
        + season_effect
        + 1000 * (df["brand_strength"] ** 4)
        + 600 * df["noise_cancellation"]
        + 950 * np.log1p(df["battery_life"]) / np.log1p(40)
        + np.random.normal(0, 150, n)
    )

    demand = np.maximum(demand, 0)

    df["demand"] = np.round(demand)

    #REVENUE

    df["revenue"] = (
        df["price"]
        * df["demand"]
    ).round(2)

    return df


