import pandas as pd
import random
import uuid
import os

def generate_synthetic_data(num_records=1000, output_path="dataset/implicit_failures.csv"):
    """
    Generate synthetic implicit failure events.
    Recovery probability is now a COMBINED score of:
      1. Current event state (failure type, amount, intent, recency)
      2. User history (payment reliability, churn risk, order frequency)
    """
    modes = ['ABANDONED_CHECKOUT', 'INCOMPLETE_PENDING', 'OVERDUE_INVOICE', 'PARTIAL_PAYMENT']
    devices = ['Mobile', 'Desktop', 'Tablet']
    times_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']

    # Load user history if it exists for combined scoring
    history_path = "dataset/user_history.csv"
    user_history = {}
    if os.path.exists(history_path):
        hist_df = pd.read_csv(history_path)
        user_history = hist_df.set_index('user_id').to_dict('index')
        print(f"Loaded {len(user_history)} user profiles for combined scoring")
    else:
        print("Warning: No user_history.csv found. Run generate_user_history.py first for combined scoring.")

    # Use user IDs from history if available, otherwise generate randomly
    available_users = list(user_history.keys()) if user_history else [f"user_{i}" for i in range(1000, 10000)]

    data = []

    for _ in range(num_records):
        event_id = str(uuid.uuid4())
        user_id = random.choice(available_users)
        mode = random.choice(modes)
        amount_at_risk = round(random.uniform(10.0, 5000.0), 2)
        time_since_last_action = random.randint(1, 48 * 60)  # 1 minute to 48 hours
        user_intent_score = round(random.uniform(0.1, 1.0), 2)
        device_type = random.choices(devices, weights=[0.6, 0.3, 0.1])[0]
        time_of_day = random.choice(times_of_day)
        is_valuable_customer = random.choices([True, False], weights=[0.2, 0.8])[0]

        # ============================================
        # COMBINED RECOVERY PROBABILITY
        # ============================================

        # --- Component 1: Current Event Score (0 to ~1.0) ---
        event_score = user_intent_score * 0.3

        if amount_at_risk < 500:
            event_score += 0.15
        elif amount_at_risk > 3000:
            event_score -= 0.05

        if time_since_last_action < 60:
            event_score += 0.25
        elif time_since_last_action < 1440:
            event_score += 0.10

        if mode == 'ABANDONED_CHECKOUT':
            event_score += 0.10
        elif mode == 'PARTIAL_PAYMENT':
            event_score += 0.08  # already showed willingness to pay
        elif mode == 'OVERDUE_INVOICE':
            event_score -= 0.05  # passive, harder to recover

        # --- Component 2: User History Score (0 to ~1.0) ---
        history_score = 0.0
        if user_id in user_history:
            h = user_history[user_id]
            total = h.get('total_orders', 0)
            success = h.get('successful_payments', 0)
            churn = h.get('churn_risk_score', 0.5)
            recency = h.get('last_successful_payment_days', 180)

            # Payment reliability boosts recovery
            reliability = success / max(total, 1)
            history_score += reliability * 0.25

            # Low churn = higher chance of coming back
            history_score += (1 - churn) * 0.20

            # Recent activity = engaged user
            if recency < 30:
                history_score += 0.15
            elif recency < 90:
                history_score += 0.08

            # Frequent buyers recover more
            if total > 20:
                history_score += 0.10
            elif total > 5:
                history_score += 0.05
        else:
            # Unknown user — use neutral history score
            history_score = 0.15

        # --- Combined Score ---
        # 55% weight on current event, 45% on user history
        combined_score = (event_score * 0.55) + (history_score * 0.45)

        # Add noise
        combined_score += random.uniform(-0.08, 0.08)

        target_recovered = 1 if combined_score >= 0.35 else 0

        data.append({
            'event_id': event_id,
            'user_id': user_id,
            'revenue_loss_mode': mode,
            'amount_at_risk': amount_at_risk,
            'time_since_last_action': time_since_last_action,
            'user_intent_score': user_intent_score,
            'device_type': device_type,
            'time_of_day': time_of_day,
            'is_valuable_customer': is_valuable_customer,
            'target_recovered': target_recovered
        })

    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_records} records at {output_path}")

    print("\nTarget Class Distribution:")
    print(df['target_recovered'].value_counts(normalize=True))

if __name__ == "__main__":
    generate_synthetic_data()
