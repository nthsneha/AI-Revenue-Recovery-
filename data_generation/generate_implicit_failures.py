import pandas as pd
import random
import uuid
import os

def generate_synthetic_data(num_records=1000, output_path="dataset/implicit_failures.csv"):
    modes = ['ABANDONED_CHECKOUT', 'INCOMPLETE_PENDING', 'OVERDUE_INVOICE', 'PARTIAL_PAYMENT']
    devices = ['Mobile', 'Desktop', 'Tablet']
    times_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
    
    data = []
    
    for _ in range(num_records):
        event_id = str(uuid.uuid4())
        user_id = f"user_{random.randint(1000, 9999)}"
        mode = random.choice(modes)
        amount_at_risk = round(random.uniform(10.0, 5000.0), 2)
        time_since_last_action = random.randint(1, 48 * 60) # 1 minute to 48 hours
        user_intent_score = round(random.uniform(0.1, 1.0), 2)
        device_type = random.choices(devices, weights=[0.6, 0.3, 0.1])[0]
        time_of_day = random.choice(times_of_day)
        is_valuable_customer = random.choices([True, False], weights=[0.2, 0.8])[0]
        
        # Heuristics for the 'target_recovered' label so the model can learn something
        # Higher intent, lower amount, and shorter time since action = higher chance of recovery
        recovery_prob = user_intent_score * 0.4
        if amount_at_risk < 500:
            recovery_prob += 0.2
        if time_since_last_action < 60:
            recovery_prob += 0.3
        elif time_since_last_action < 1440:
            recovery_prob += 0.1
            
        if mode == 'ABANDONED_CHECKOUT':
            recovery_prob += 0.1
            
        # Add some noise
        recovery_prob += random.uniform(-0.1, 0.1)
        
        target_recovered = 1 if recovery_prob >= 0.55 else 0
        
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
    
    # Print a quick distribution check
    print("\nTarget Class Distribution:")
    print(df['target_recovered'].value_counts(normalize=True))

if __name__ == "__main__":
    generate_synthetic_data()
