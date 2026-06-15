"""
Adaptive Revenue Engine (ARE)
Product : Wireless Bluetooth Earbuds
Market Environment Simulator

Creates a synthetic market environment of the product
for causal inference and reinforment learning.
"""

import numpy as np

def market_environment(price, noise_c, battery, brand, season):

    if season == 0:
        season_effect = np.random.uniform(-420, -120)

    elif season == 1:
        season_effect = np.random.uniform(-100, 100)

    else:
        season_effect = np.random.uniform(120, 600)

    demand = (
        np.random.uniform(3000, 6000)
        - 3.2 * price
        + season_effect
        + 1000 * (brand ** 4)
        + 600 * noise_c
        + 950 * np.log1p(battery) / np.log1p(40)
        + np.random.normal(0, 150)
    )

    demand = max(0, round(demand))

    revenue = round(price * demand, 2)

    return demand, revenue