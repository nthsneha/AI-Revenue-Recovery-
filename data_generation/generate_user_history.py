"""
Generate synthetic user history profiles.

Design notes
------------
- Deterministic: seeded so every build produces the same dataset.
- Includes the demo user IDs explicitly so the demo exercises the
  history-scoring path instead of always falling back to neutral defaults.
- Profiles are generated from correlated latent traits (a "good customer"
  tends to have high order counts, high reliability AND low churn together)
  rather than each column being sampled independently. Independent columns
  make the joint distribution unrealistic and let a tree model separate
  classes far too easily.
"""

import os
import random

import numpy as np
import pandas as pd

SEED = 42

# User IDs referenced by the demo logs in main.py. Forced into the dataset so
# the viability gate demonstrates real history lookup, not neutral defaults.
DEMO_USER_IDS = ["user_8231", "user_1120", "user_5531"]


def generate_user_history(num_users=5000, output_path="dataset/user_history.csv"):
    random.seed(SEED)
    np.random.seed(SEED)

    # Sample the random pool, then guarantee the demo users are present.
    pool = [f"user_{i}" for i in random.sample(range(1000, 10000), num_users)]
    user_ids = list(dict.fromkeys(DEMO_USER_IDS + pool))

    data = []

    for user_id in user_ids:
        # ------------------------------------------------------------------
        # Latent customer quality (0 = poor, 1 = excellent).
        # This drives several observable columns at once, which is what
        # creates realistic correlation between them. It is NOT written to
        # the CSV -- it is not a feature, it is the generative cause.
        # ------------------------------------------------------------------
        quality = np.clip(np.random.beta(2.2, 2.2), 0.01, 0.99)

        # Engagement level: better customers order more, with heavy tail.
        total_orders = int(np.random.gamma(shape=1.5 + 3.0 * quality, scale=8.0))
        total_orders = min(total_orders, 200)

        # Payment reliability tracks quality, with noise.
        success_rate = np.clip(np.random.normal(0.55 + 0.40 * quality, 0.10), 0.0, 1.0)
        successful_payments = int(round(total_orders * success_rate))
        failed_payments = total_orders - successful_payments

        # Spend is lognormal (realistic: right-skewed), loosely tied to quality.
        avg_order_value = float(np.round(
            np.exp(np.random.normal(5.6 + 0.6 * quality, 0.75)), 2
        ))
        avg_order_value = float(np.clip(avg_order_value, 20.0, 25000.0))

        # Recency: engaged customers bought recently. Exponential-ish tail.
        recency_scale = 220.0 - 170.0 * quality
        last_successful_payment_days = int(np.clip(
            np.random.exponential(recency_scale), 0, 365
        ))
        if total_orders == 0:
            last_successful_payment_days = 365

        # Preferred payment / channel shift with quality (subscribers and
        # engaged users skew toward UPI + WhatsApp in this market).
        pay_weights = [0.25 + 0.20 * quality, 0.25, 0.20, 0.15 - 0.05 * quality,
                       0.15 - 0.05 * quality]
        pay_weights = [max(w, 0.01) for w in pay_weights]
        preferred_payment = random.choices(
            ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"],
            weights=pay_weights,
        )[0]

        chan_weights = [0.30 + 0.20 * quality, 0.30, 0.20 - 0.05 * quality,
                        0.20 - 0.05 * quality]
        chan_weights = [max(w, 0.01) for w in chan_weights]
        preferred_channel = random.choices(
            ["WhatsApp", "Email", "SMS", "Push Notification"],
            weights=chan_weights,
        )[0]

        # Churn risk derived from behaviour, then noised. Deliberately an
        # imperfect proxy for `quality` so it isn't a giveaway column.
        if total_orders == 0:
            churn_risk = np.random.uniform(0.65, 1.0)
        else:
            reliability = successful_payments / max(total_orders, 1)
            recency_factor = min(last_successful_payment_days / 365.0, 1.0)
            churn_risk = (
                (1 - reliability) * 0.35
                + recency_factor * 0.40
                + (1 - quality) * 0.15
                + np.random.normal(0, 0.09)
            )
        churn_risk = float(np.round(np.clip(churn_risk, 0.0, 1.0), 2))

        is_subscribed = random.random() < (0.05 + 0.25 * quality)

        data.append({
            "user_id": user_id,
            "total_orders": total_orders,
            "successful_payments": successful_payments,
            "failed_payments": failed_payments,
            "avg_order_value": avg_order_value,
            "preferred_payment_method": preferred_payment,
            "preferred_channel": preferred_channel,
            "last_successful_payment_days": last_successful_payment_days,
            "churn_risk_score": churn_risk,
            "is_subscribed": is_subscribed,
        })

    df = pd.DataFrame(data)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} user profiles at {output_path}")
    print(f"Demo users included: {[u for u in DEMO_USER_IDS if u in set(df.user_id)]}")
    print("\nSample profiles:")
    print(df.head())
    print("\nChurn risk distribution:")
    print(df["churn_risk_score"].describe())
    print("\nTotal orders distribution:")
    print(df["total_orders"].describe())


if __name__ == "__main__":
    generate_user_history()