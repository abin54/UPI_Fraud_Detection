"""
Synthetic UPI Transaction Data Generator
Generates realistic Indian UPI transaction data for fraud detection modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Indian banks and UPI handles
BANKS = ['sbi', 'hdfc', 'icici', 'axis', 'kotak', 'pnb', 'bob', 'canara', 'idbi', 'yes']
UPI_APPS = ['paytm', 'gpay', 'phonepe', 'bhim', 'amazonpay', 'whatsapp']
MERCHANT_CATEGORIES = [
    'grocery', 'restaurant', 'fuel', 'electronics', 'clothing',
    'pharmacy', 'utilities', 'entertainment', 'travel', 'education',
    'recharge', 'money_transfer', 'bills', 'insurance', 'investment'
]

# Indian city names
CITIES = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata',
    'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Bhopal', 'Patna',
    'Indore', 'Nagpur', 'Surat', 'Vadodara', 'Coimbatore', 'Kochi'
]

# Device types
DEVICES = ['android', 'ios']
DEVICE_MODELS = {
    'android': ['Samsung Galaxy', 'Xiaomi Redmi', 'OnePlus', 'Realme', 'Vivo', 'Oppo', 'Poco'],
    'ios': ['iPhone 12', 'iPhone 13', 'iPhone 14', 'iPhone 11', 'iPhone SE']
}

def generate_upi_handle(is_merchant=False):
    """Generate realistic Indian UPI handle"""
    if is_merchant:
        merchant_names = ['bigbazaar', 'dmart', 'reliance', 'swiggy', 'zomato',
                         'flipkart', 'amazon', 'myntra', 'ola', 'uber']
        return f"{random.choice(merchant_names)}.merchant@{random.choice(BANKS)}"
    else:
        first_names = ['rahul', 'priya', 'amit', 'sneha', 'vijay', 'pooja',
                      'raj', 'anjali', 'deepak', 'kavita', 'suresh', 'neha']
        return f"{random.choice(first_names)}{random.randint(100, 9999)}@{random.choice(UPI_APPS)}"

def generate_device_fingerprint():
    """Generate device fingerprint"""
    device_type = random.choice(DEVICES)
    model = random.choice(DEVICE_MODELS[device_type])
    return {
        'device_type': device_type,
        'device_model': model,
        'device_id': ''.join(random.choices('abcdef0123456789', k=16)),
        'app_version': f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}"
    }

def generate_transaction(user_id, user_profile, is_fraud=False):
    """Generate a single transaction"""

    # Base transaction time
    base_date = datetime(2024, 1, 1)
    days_offset = random.randint(0, 365)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    txn_time = base_date + timedelta(days=days_offset, hours=hour, minutes=minute)

    # Normal transaction patterns
    if not is_fraud:
        # Normal amount based on category
        category = random.choice(MERCHANT_CATEGORIES)
        if category in ['grocery', 'restaurant', 'recharge']:
            amount = np.random.lognormal(5, 1)  # Small amounts
        elif category in ['electronics', 'travel', 'insurance']:
            amount = np.random.lognormal(8, 1.5)  # Large amounts
        else:
            amount = np.random.lognormal(6, 1.2)  # Medium amounts

        amount = min(max(amount, 10), 100000)  # Cap between 10 and 1 lakh

        # Use user's regular device
        device = user_profile['primary_device']
        location = user_profile['home_city']

        # Normal hours (6 AM to 11 PM mostly)
        if hour < 6 or hour > 23:
            if random.random() > 0.1:  # 90% chance to shift to normal hours
                hour = random.randint(8, 21)
                txn_time = txn_time.replace(hour=hour)

    else:  # Fraud patterns
        fraud_type = random.choice(['high_amount', 'velocity', 'new_device', 'unusual_time', 'distant_location'])

        category = random.choice(MERCHANT_CATEGORIES)

        if fraud_type == 'high_amount':
            # Unusually high amount
            amount = np.random.uniform(50000, 200000)
        elif fraud_type == 'velocity':
            # Multiple transactions in short time (handled in batch)
            amount = np.random.uniform(1000, 20000)
        else:
            amount = np.random.lognormal(7, 1.5)

        # New/different device for fraud
        if fraud_type == 'new_device' or random.random() > 0.6:
            device = generate_device_fingerprint()
        else:
            device = user_profile['primary_device']

        # Different location
        if fraud_type == 'distant_location' or random.random() > 0.7:
            location = random.choice([c for c in CITIES if c != user_profile['home_city']])
        else:
            location = user_profile['home_city']

        # Unusual time
        if fraud_type == 'unusual_time':
            hour = random.choice([0, 1, 2, 3, 4, 5])
            txn_time = txn_time.replace(hour=hour)

    return {
        'transaction_id': f"TXN{random.randint(100000000, 999999999)}",
        'user_id': user_id,
        'sender_upi': user_profile['upi_handle'],
        'receiver_upi': generate_upi_handle(is_merchant=random.random() > 0.3),
        'amount': round(amount, 2),
        'currency': 'INR',
        'transaction_type': random.choice(['P2P', 'P2M', 'P2M']),  # More P2M
        'merchant_category': category,
        'timestamp': txn_time,
        'hour': txn_time.hour,
        'day_of_week': txn_time.weekday(),
        'is_weekend': txn_time.weekday() >= 5,
        'device_type': device['device_type'],
        'device_model': device['device_model'],
        'device_id': device['device_id'],
        'app_version': device['app_version'],
        'location_city': location,
        'is_fraud': int(is_fraud)
    }

def generate_user_profile(user_id):
    """Generate a user profile"""
    return {
        'user_id': user_id,
        'upi_handle': generate_upi_handle(),
        'home_city': random.choice(CITIES),
        'primary_device': generate_device_fingerprint(),
        'account_age_days': random.randint(30, 1500),
        'avg_monthly_txns': random.randint(5, 100)
    }

def generate_dataset(n_users=1000, n_transactions=50000, fraud_ratio=0.02):
    """Generate complete synthetic dataset"""

    print(f"Generating {n_transactions} transactions for {n_users} users...")
    print(f"Fraud ratio: {fraud_ratio*100}%")

    # Generate user profiles
    users = {f"USER_{i:05d}": generate_user_profile(f"USER_{i:05d}")
             for i in range(n_users)}

    transactions = []
    n_fraud = int(n_transactions * fraud_ratio)
    n_normal = n_transactions - n_fraud

    # Generate normal transactions
    print("Generating normal transactions...")
    for _ in range(n_normal):
        user_id = random.choice(list(users.keys()))
        txn = generate_transaction(user_id, users[user_id], is_fraud=False)
        transactions.append(txn)

    # Generate fraud transactions
    print("Generating fraud transactions...")
    for _ in range(n_fraud):
        user_id = random.choice(list(users.keys()))
        txn = generate_transaction(user_id, users[user_id], is_fraud=True)
        transactions.append(txn)

    # Create DataFrame
    df = pd.DataFrame(transactions)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\nDataset generated successfully!")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Unique users: {df['user_id'].nunique()}")

    return df, users

def save_dataset(df, users, output_dir='../data'):
    """Save generated dataset"""
    os.makedirs(output_dir, exist_ok=True)

    # Save transactions
    df.to_csv(f'{output_dir}/synthetic_upi_transactions.csv', index=False)
    print(f"Saved transactions to {output_dir}/synthetic_upi_transactions.csv")

    # Save user profiles
    user_df = pd.DataFrame(users).T
    user_df.to_csv(f'{output_dir}/user_profiles.csv')
    print(f"Saved user profiles to {output_dir}/user_profiles.csv")

if __name__ == "__main__":
    # Generate dataset
    df, users = generate_dataset(
        n_users=1000,
        n_transactions=50000,
        fraud_ratio=0.02  # 2% fraud rate
    )

    # Save dataset
    save_dataset(df, users)

    # Display sample
    print("\nSample transactions:")
    print(df.head(10))

    print("\nColumn info:")
    print(df.dtypes)
