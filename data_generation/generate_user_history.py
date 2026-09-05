import pandas as pd
import random
import os

def generate_user_history(num_users=300, output_path="dataset/user_history.csv"):
    """
    Generate synthetic user history profiles.
    Uses user IDs from user_1000 to user_9999 (same pool as implicit_failures).
    """
    # Use a fixed pool of user IDs so there's overlap with the events dataset
    user_ids = [f"user_{i}" for i in random.sample(range(1000, 10000), num_users)]

    data = []

    for user_id in user_ids:
        total_orders = random.randint(0, 120)
        # Successful payments are a proportion of total orders
        success_rate = random.uniform(0.5, 1.0)
        successful_payments = int(total_orders * success_rate)
        failed_payments = total_orders - successful_payments

        avg_order_value = round(random.uniform(50.0, 5000.0), 2)
        preferred_payment = random.choices(
            ['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Wallet'],
            weights=[0.35, 0.25, 0.20, 0.10, 0.10]
        )[0]
        preferred_channel = random.choices(
            ['WhatsApp', 'Email', 'SMS', 'Push Notification'],
            weights=[0.40, 0.30, 0.15, 0.15]
        )[0]
        last_successful_payment_days = random.randint(0, 365)

        # Churn risk derived from behavior signals
        churn_risk = 0.0
        if total_orders == 0:
            churn_risk = round(random.uniform(0.7, 1.0), 2)
        else:
            payment_reliability = successful_payments / max(total_orders, 1)
            recency_factor = min(last_successful_payment_days / 365.0, 1.0)
            churn_risk = round((1 - payment_reliability) * 0.4 + recency_factor * 0.4 + random.uniform(0, 0.2), 2)
            churn_risk = min(churn_risk, 1.0)

        is_subscribed = random.choices([True, False], weights=[0.15, 0.85])[0]

        data.append({
            'user_id': user_id,
            'total_orders': total_orders,
            'successful_payments': successful_payments,
            'failed_payments': failed_payments,
            'avg_order_value': avg_order_value,
            'preferred_payment_method': preferred_payment,
            'preferred_channel': preferred_channel,
            'last_successful_payment_days': last_successful_payment_days,
            'churn_risk_score': churn_risk,
            'is_subscribed': is_subscribed
        })

    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_users} user profiles at {output_path}")

    print("\nSample profiles:")
    print(df.head())
    print(f"\nChurn risk distribution:")
    print(df['churn_risk_score'].describe())

if __name__ == "__main__":
    generate_user_history()
