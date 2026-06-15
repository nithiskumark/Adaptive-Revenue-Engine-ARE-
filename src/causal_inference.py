"""
Adaptive Revenue Engine (ARE)
Causal Engine
"""


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from dowhy import CausalModel

from data_generate import gen_data

# DATA VALIDATION
def validate_data(df):

    required_cols = [
        "price",
        "demand",
        "season",
        "noise_cancellation",
        "brand_strength",
        "battery_life"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if len(missing) > 0:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    return True

# BUILD CAUSAL MODEL
def build_causal_model(df):

    model = CausalModel(
        data=df,
        treatment="price",
        outcome="demand",
        common_causes=[
            "season",
            "noise_cancellation",
            "brand_strength",
            "battery_life"
        ]
    )

    return model

# IDENTIFY ESTIMATE VALUE
def identify_estimate(model):

    identified_estimand = model.identify_effect()

    estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
    )

    return identified_estimand, estimate

# REFUTATION TESTS
def run_refutations(
    model,
    identified_estimand,
    estimate
):

    results = {}

    try:

        placebo = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter"
        )

        results["placebo"] = placebo

    except Exception as e:

        results["placebo"] = str(e)

    try:

        random_common = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="random_common_cause"
        )

        results["random_common_cause"] = random_common

    except Exception as e:

        results["random_common_cause"] = str(e)

    return results

def run_causal_analysis(df: pd.DataFrame | None = None) -> dict:

    if df is None:
        df = gen_data()

    validate_data(df)

    print("Building causal model...")
    model = build_causal_model(df)

    print("Identifying and estimating causal effect...")
    estimand, estimate = identify_estimate(model)

    print(f"\nEstimated ATE of price on demand: {estimate.value:.4f}")
    print("(Interpretation: a $1 increase in price changes demand by this amount)\n")

    print("Running refutation tests...")
    refutations = run_refutations(model, estimand, estimate)

    for name, result in refutations.items():
        print(f"\n[{name}]")
        print(result)

    return {
        "model": model,
        "estimand": estimand,
        "estimate": estimate,
        "refutations": refutations,
    }


if __name__ == "__main__":
    run_causal_analysis()


