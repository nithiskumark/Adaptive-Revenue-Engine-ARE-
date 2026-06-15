"""
Adaptive Revenue Engine (ARE)
Reinforment Learning
Contextual Bandit Pricing Agent
"""

import numpy as np
import pandas as pd
import os
import pickle

from data_generate import gen_data

# FEATURES
def baseline_price(df):
    
    # Historical average price.
    
    return float(df["price"].mean()) 

def bucket_battery(battery):

    # Battery segmentation.


    if battery < 10:
        return "low"

    elif battery < 18:
        return "medium"

    return "high"

def bucket_brand(brand):

    if brand < 0.30:
        return "weak"

    elif brand < 0.70:
        return "mid"

    return "strong"

def create_price_grid(
        min_price=90,
        max_price=1500,
        n_prices=25
):
    
    # Action space.
    

    return np.round(
        np.linspace(
            min_price,
            max_price,
            n_prices
        ),
        2
    )

# AGENT
class PricingAgent:

    """
    Contextual Multi-Armed Bandit Pricing Agent

    State:
        (
            season,
            battery_bucket,
            brand_bucket,
            noise
        )

    Actions:
        Price levels

    Reward:
        Revenue
    """

    def __init__(
            self,
            prices,
            epsilon=0.10
    ):

        self.prices = list(prices)
        self.epsilon = epsilon

        self.q = {}
        self.count = {}

        self.total_reward = 0
        self.total_steps = 0

    # STATE

    def get_state(
            self,
            season,
            battery,
            brand,
            noise
    ):

        return (
            season,
            bucket_battery(battery),
            bucket_brand(brand),
            noise
        )
    
    # INIT STATE

    def init_state(self, state):

        if state not in self.q:

            self.q[state] = {
                p: 0.0
                for p in self.prices
            }

            self.count[state] = {
                p: 0
                for p in self.prices
            }

    # ACTION

    def choose(self, state):

        self.init_state(state)

        # Exploration

        if np.random.rand() < self.epsilon:

            return float(
                np.random.choice(self.prices)
            )

        # Exploitation

        return max(
            self.q[state],
            key=self.q[state].get
        )
    
    # UPDATE

    def update(
            self,
            state,
            price,
            reward
    ):

        self.init_state(state)

        self.count[state][price] += 1

        n = self.count[state][price]

        q_old = self.q[state][price]

        self.q[state][price] += (
            reward - q_old
        ) / n

        self.total_reward += reward
        self.total_steps += 1

    # BEST ACTION 

    def best_price(self, state):

        self.init_state(state)

        return max(
            self.q[state],
            key=self.q[state].get
        )
    
    # Expected Revenue

    def expected_revenue(self, state):

        self.init_state(state)

        best_price = self.best_price(state)

        return self.q[state][best_price]

    # SAVE THE AGENT

    def save_agent(
            agent,
            path="models/pricing_agent.pkl"
    ):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        with open(path, "wb") as f:

            pickle.dump(agent, f)
        
        print(f"Agent saved → {path}")


