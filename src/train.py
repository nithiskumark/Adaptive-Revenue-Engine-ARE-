"""
Adaptive Revenue Engine (ARE)
Train Reinforment Learning
Contextual Bandit Pricing Agent
"""


import os
import json
import pickle
import numpy as np
import pandas as pd

from environment import market_environment
from rl_agent import PricingAgent, create_price_grid

# CONFIG

N_EPISODES = 10000

PRICE_GRID = np.linspace(90, 1500, 25)

EPSILON = 0.12

MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)

# TRAIN RL AGENT
def train_agent():

    print("=" * 60)
    print("ADAPTIVE REVENUE ENGINE TRAINING")
    print("=" * 60)

    agent = PricingAgent(
        prices=PRICE_GRID,
        epsilon=EPSILON
    )

    rewards = []

    episode_logs = []

    best_revenue = -np.inf

    for episode in range(N_EPISODES):

        season = np.random.choice([0, 1, 2])

        noise_c = np.random.choice([0, 1])

        battery = np.random.uniform(4, 24)

        brand = np.random.uniform(0, 1)

        state = agent.get_state(
            season,
            battery,
            brand,
            noise_c
        )

        price = agent.choose(state)

        demand, revenue = market_environment(
            price=price,
            noise_c=noise_c,
            battery=battery,
            brand=brand,
            season=season
        )

        agent.update(
            state,
            price,
            revenue
        )

        rewards.append(revenue)

        episode_logs.append(
            {
                "episode": episode,
                "season": season,
                "noise": noise_c,
                "battery": battery,
                "brand": brand,
                "price": price,
                "demand": demand,
                "revenue": revenue
            }
        )

        best_revenue = max(best_revenue, revenue)

        if episode % 500 == 0:

            avg_reward = np.mean(
                rewards[max(0, len(rewards)-500):]
            )

            print(
                f"Episode={episode:5d} | "
                f"Avg Revenue={avg_reward:10.2f} | "
                f"Best Revenue={best_revenue:10.2f}"
            )

    print("\nTraining Complete")

    return agent, rewards, episode_logs

def save_logs(episode_logs, path="../models/episode_logs.csv"):
    df = pd.DataFrame(episode_logs)
    df.to_csv(path, index=False)
    print(f"Episode logs saved → {path}")


def save_rewards(rewards, path="../models/rewards.csv"):
    pd.DataFrame({"episode": range(len(rewards)), "revenue": rewards}).to_csv(
        path, index=False
    )
    print(f"  Rewards CSV saved → {path}")


if __name__ == "__main__":
    agent, rewards, episode_logs = train_agent()

    model_path = os.path.join(MODEL_DIR, "pricing_agent.pkl")
    agent.save_agent(model_path)

    save_logs(episode_logs, os.path.join(MODEL_DIR, "episode_logs.csv"))
    save_rewards(rewards, os.path.join(MODEL_DIR, "rewards.csv"))

    print("\nAll artifacts saved. Done.")




























