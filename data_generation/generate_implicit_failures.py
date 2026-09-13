"""
Generate synthetic revenue-loss events with a *probabilistic* recovery label.

Why this differs from a naive generator
---------------------------------------
The previous version computed a score from the same columns it exported as
features, then thresholded it into a hard 0/1 label. That makes the label a
deterministic function of the features, so any model simply reverse-engineers
the scoring rules -- accuracy looks excellent and means nothing.

This version fixes that in four ways:

1. LATENT CONFOUNDERS. Recovery is partly driven by factors that are never
   exported as features (a hidden per-user responsiveness trait, whether a
   competitor was running a promo, whether the payment channel was degraded).
   These create genuine irreducible error -- a Bayes error floor.

2. PROBABILISTIC LABELS. The score becomes a probability via a logistic link
   and the outcome is *sampled*. Two identical feature rows can therefore
   have different outcomes, which is how real conversion data behaves.

3. NON-LINEAR STRUCTURE. Log decay on recency, a non-monotonic amount effect,
   and genuine interaction terms -- so a tree ensemble has something to learn
   that a linear model cannot trivially match.

4. LABEL NOISE. A small fraction of outcomes are flipped, mimicking tracking
   errors and mislabelled outcomes in real pipelines.

The exported feature schema is intentionally UNCHANGED so it stays compatible
with the inference path in agent/graph.py.
"""

import hashlib
import math
import os
import random

import numpy as np
import pandas as pd

SEED = 42

# Salt for the deterministic latent trait. Because this is derived by hashing
# rather than stored, the trait is reproducible across scripts but never
# appears in the dataset -- it cannot leak into the feature matrix.
LATENT_SALT = "revenue-recovery-latent-v1"

MODES = [
    "ABANDONED_CHECKOUT",
    "INCOMPLETE_PENDING",
    "OVERDUE_INVOICE",
    "PARTIAL_PAYMENT",
    "PAYMENT_FAILURE",   # was missing before; triage can emit it, so train on it
]
MODE_WEIGHTS = [0.28, 0.14, 0.18, 0.12, 0.28]

DEVICES = ["Mobile", "Desktop", "Tablet"]
TIMES_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]

# Per-mode log-odds shift. Active-intent failures recover better than passive
# ones; an overdue invoice is the hardest because the user has already lapsed.
MODE_EFFECT = {
    "ABANDONED_CHECKOUT": 0.30,
    "PAYMENT_FAILURE": 0.55,    # user tried to pay -- strong intent signal
    "PARTIAL_PAYMENT": 0.35,
    "INCOMPLETE_PENDING": -0.10,
    "OVERDUE_INVOICE": -0.55,
}

LABEL_NOISE_RATE = 0.06   # fraction of outcomes randomly flipped
INTERCEPT = -0.55         # tunes the overall base rate


def latent_responsiveness(user_id: str) -> float:
    """
    A hidden per-user propensity to respond to recovery outreach.

    Deterministic given the user_id, so it is stable across runs and across
    scripts, but it is never written to disk as a column. Uses hashlib rather
    than the builtin hash() because string hashing is salted per process.
    """
    digest = hashlib.md5(f"{LATENT_SALT}:{user_id}".encode()).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return unit  # in [0, 1]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_synthetic_data(num_records=50000,
                            output_path="dataset/implicit_failures.csv",
                            history_path="dataset/user_history.csv"):
    random.seed(SEED)
    np.random.seed(SEED)

    user_history = {}
    if os.path.exists(history_path):
        hist_df = pd.read_csv(history_path)
        user_history = hist_df.set_index("user_id").to_dict("index")
        print(f"Loaded {len(user_history)} user profiles for combined scoring")
    else:
        print("Warning: no user_history.csv found. "
              "Run generate_user_history.py first for combined scoring.")

    available_users = (list(user_history.keys()) if user_history
                       else [f"user_{i}" for i in range(1000, 10000)])

    rows = []
    true_probs = []

    for _ in range(num_records):
        user_id = random.choice(available_users)
        mode = random.choices(MODES, weights=MODE_WEIGHTS)[0]

        # Amount is lognormal -- most events are small, a few are large.
        amount_at_risk = float(np.round(
            np.clip(np.exp(np.random.normal(6.0, 1.15)), 10.0, 50000.0), 2
        ))

        # Recency is heavily right-skewed: most events are recent.
        time_since_last_action = int(np.clip(
            np.random.exponential(240.0), 1, 48 * 60
        ))

        user_intent_score = float(np.round(
            np.clip(np.random.beta(2.5, 2.0), 0.01, 1.0), 2
        ))
        device_type = random.choices(DEVICES, weights=[0.62, 0.28, 0.10])[0]
        time_of_day = random.choice(TIMES_OF_DAY)

        h = user_history.get(user_id, {})
        total_orders = h.get("total_orders", 0)
        successful = h.get("successful_payments", 0)
        churn = h.get("churn_risk_score", 0.5)
        recency_days = h.get("last_successful_payment_days", 180)
        avg_order_value = h.get("avg_order_value", 500.0)
        reliability = successful / max(total_orders, 1)

        # "Valuable" is a coarse, deliberately noisy business flag -- it is a
        # weak proxy, not a clean function of spend.
        is_valuable_customer = bool(
            (avg_order_value > 1500 and total_orders > 10)
            != (random.random() < 0.12)   # 12% of the flags are just wrong
        )

        # ------------------------------------------------------------------
        # LATENT FACTORS -- influence the outcome, never exported as features.
        # These are what stop the label from being a pure function of X.
        # ------------------------------------------------------------------
        responsiveness = latent_responsiveness(user_id)      # hidden user trait
        channel_degraded = random.random() < 0.08            # gateway/UPI issues
        competitor_promo = random.random() < 0.15            # lost to a rival
        payday_window = random.random() < 0.22               # user has cash now

        # ------------------------------------------------------------------
        # LOG-ODDS MODEL (non-linear, with interactions)
        # ------------------------------------------------------------------
        z = INTERCEPT

        # Intent: strong, roughly linear driver.
        z += 1.45 * (user_intent_score - 0.5)

        # Recency: log decay, not a step function. A 10-minute-old event is
        # far more recoverable than a 10-hour-old one; 24h vs 36h barely differs.
        z += -0.42 * math.log1p(time_since_last_action / 60.0)

        # Amount: non-monotonic. Very small amounts are trivially recoverable,
        # mid amounts are the sweet spot, very large ones stall on approval.
        amt_log = math.log10(max(amount_at_risk, 1.0))
        z += 0.55 * amt_log - 0.16 * (amt_log ** 2)

        # Failure mode.
        z += MODE_EFFECT[mode]

        # History effects.
        z += 1.30 * (reliability - 0.5)
        z += -1.10 * (churn - 0.5)
        z += -0.30 * math.log1p(recency_days / 30.0)
        z += 0.22 * math.log1p(total_orders) / 2.0

        # --- Genuine interactions (what justifies a tree ensemble) ---
        # High intent only converts if the customer is actually reliable.
        z += 1.05 * (user_intent_score - 0.5) * (reliability - 0.5) * 2.0
        # Large amounts hurt much more for churn-risky users.
        z += -0.60 * max(amt_log - 2.5, 0.0) * (churn - 0.5) * 2.0
        # Mobile users abandon large carts more often.
        if device_type == "Mobile":
            z += -0.28 * max(amt_log - 2.8, 0.0)
        # Night-time outreach performs worse.
        if time_of_day == "Night":
            z += -0.22

        # --- Latent contributions (unobservable => Bayes error) ---
        z += 1.20 * (responsiveness - 0.5)
        z += -0.85 if channel_degraded else 0.0
        z += -0.70 if competitor_promo else 0.0
        z += 0.45 if payday_window else 0.0

        # Idiosyncratic per-event noise.
        z += np.random.normal(0, 0.45)

        p = sigmoid(z)
        target = 1 if random.random() < p else 0

        # Label noise: tracking failures, misattributed conversions.
        if random.random() < LABEL_NOISE_RATE:
            target = 1 - target

        true_probs.append(p)
        rows.append({
            "event_id": f"evt_{len(rows):07d}",
            "user_id": user_id,
            "revenue_loss_mode": mode,
            "amount_at_risk": amount_at_risk,
            "time_since_last_action": time_since_last_action,
            "user_intent_score": user_intent_score,
            "device_type": device_type,
            "time_of_day": time_of_day,
            "is_valuable_customer": is_valuable_customer,
            "target_recovered": target,
        })

    df = pd.DataFrame(rows)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} events at {output_path}")
    print("\nTarget class distribution:")
    print(df["target_recovered"].value_counts(normalize=True).round(4))
    print("\nEvents per mode:")
    print(df["revenue_loss_mode"].value_counts())

    tp = np.array(true_probs)
    bayes_acc = np.mean(np.maximum(tp, 1 - tp)) * (1 - LABEL_NOISE_RATE) + \
                np.mean(np.minimum(tp, 1 - tp)) * LABEL_NOISE_RATE
    print(f"\nApprox. Bayes-optimal accuracy ceiling: {bayes_acc:.3f}")
    print("Anything close to this means the model has learned all the "
          "learnable signal; anything at ~1.00 would mean the label leaked.")


if __name__ == "__main__":
    generate_synthetic_data()