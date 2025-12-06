"""
Feature Engineering for UPI Fraud Detection
Creates domain-specific features for fraud detection
"""

import pandas as pd
import numpy as np
from datetime import timedelta

class FraudFeatureEngineer:
    """Feature engineering pipeline for fraud detection"""

    def __init__(self):
        self.merchant_risk_scores = {}
        self.user_statistics = {}

    def fit(self, df):
        """Calculate statistics from training data"""
        print("Calculating merchant risk scores...")
        self._calculate_merchant_risk(df)

        print("Calculating user statistics...")
        self._calculate_user_stats(df)

        return self

    def _calculate_merchant_risk(self, df):
        """Calculate fraud rate per merchant category"""
        fraud_rates = df.groupby('merchant_category')['is_fraud'].mean()
        # Normalize to 0-1 scale
        self.merchant_risk_scores = (fraud_rates / fraud_rates.max()).to_dict()

    def _calculate_user_stats(self, df):
        """Calculate per-user statistics"""
        user_stats = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'max'],
            'transaction_id': 'count'
        }).reset_index()
        user_stats.columns = ['user_id', 'avg_amount', 'std_amount', 'max_amount', 'txn_count']
        self.user_statistics = user_stats.set_index('user_id').to_dict('index')

    def transform(self, df):
        """Apply feature engineering"""
        df = df.copy()

        # Time-based features
        df = self._add_time_features(df)

        # Transaction velocity features
        df = self._add_velocity_features(df)

        # Amount-based features
        df = self._add_amount_features(df)

        # Device and location features
        df = self._add_device_location_features(df)

        # Merchant risk score
        df = self._add_merchant_risk(df)

        # User behavior deviation
        df = self._add_user_deviation_features(df)

        return df

    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)

    def _add_time_features(self, df):
        """Add time-based features"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Hour categories
        df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
        df['is_business_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)

        # Time since midnight (cyclical)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week (cyclical)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _add_velocity_features(self, df):
        """Add transaction velocity features (count and amount in time windows)"""
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Calculate rolling features per user
        velocity_features = []

        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id].copy()

            # Set timestamp as index for rolling calculations
            user_df = user_df.set_index('timestamp')

            # Transactions in last 1 hour
            user_df['txn_count_1h'] = user_df.rolling('1H')['amount'].count()

            # Transactions in last 24 hours
            user_df['txn_count_24h'] = user_df.rolling('24H')['amount'].count()

            # Total amount in last 1 hour
            user_df['amount_sum_1h'] = user_df.rolling('1H')['amount'].sum()

            # Total amount in last 24 hours
            user_df['amount_sum_24h'] = user_df.rolling('24H')['amount'].sum()

            # Average amount in last 24 hours
            user_df['amount_avg_24h'] = user_df.rolling('24H')['amount'].mean()

            user_df = user_df.reset_index()
            velocity_features.append(user_df)

        df = pd.concat(velocity_features).sort_index()

        # Fill NaN with 0 or 1 for first transactions
        df['txn_count_1h'] = df['txn_count_1h'].fillna(1)
        df['txn_count_24h'] = df['txn_count_24h'].fillna(1)
        df['amount_sum_1h'] = df['amount_sum_1h'].fillna(df['amount'])
        df['amount_sum_24h'] = df['amount_sum_24h'].fillna(df['amount'])
        df['amount_avg_24h'] = df['amount_avg_24h'].fillna(df['amount'])

        return df

    def _add_amount_features(self, df):
        """Add amount-based features"""
        # Log transform of amount
        df['amount_log'] = np.log1p(df['amount'])

        # Amount bins
        df['amount_category'] = pd.cut(
            df['amount'],
            bins=[0, 100, 500, 2000, 10000, 50000, float('inf')],
            labels=['micro', 'small', 'medium', 'large', 'very_large', 'huge']
        )

        # Is round amount (psychological pricing)
        df['is_round_amount'] = df['amount'].apply(
            lambda x: 1 if x % 100 == 0 or x % 500 == 0 or x % 1000 == 0 else 0
        )

        # Amount deviation from rolling average
        df['amount_deviation'] = (df['amount'] - df['amount_avg_24h']) / (df['amount_avg_24h'] + 1)

        return df

    def _add_device_location_features(self, df):
        """Add device and location features"""
        # Device type encoding
        df['is_ios'] = (df['device_type'] == 'ios').astype(int)

        # New device flag (first time seeing this device for user)
        df['device_user_key'] = df['user_id'] + '_' + df['device_id']
        device_first_seen = df.groupby('device_user_key')['timestamp'].transform('min')
        df['is_new_device'] = (df['timestamp'] == device_first_seen).astype(int)

        # Different city from usual (based on mode city per user)
        user_home_city = df.groupby('user_id')['location_city'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        df['home_city'] = df['user_id'].map(user_home_city)
        df['is_different_city'] = (df['location_city'] != df['home_city']).astype(int)

        # Clean up
        df = df.drop(['device_user_key', 'home_city'], axis=1)

        return df

    def _add_merchant_risk(self, df):
        """Add merchant category risk score"""
        df['merchant_risk_score'] = df['merchant_category'].map(self.merchant_risk_scores)
        df['merchant_risk_score'] = df['merchant_risk_score'].fillna(0.5)
        return df

    def _add_user_deviation_features(self, df):
        """Add features measuring deviation from user's normal behavior"""
        # Map user statistics
        df['user_avg_amount'] = df['user_id'].apply(
            lambda x: self.user_statistics.get(x, {}).get('avg_amount', df['amount'].mean())
        )
        df['user_std_amount'] = df['user_id'].apply(
            lambda x: self.user_statistics.get(x, {}).get('std_amount', df['amount'].std())
        )

        # Z-score of amount for this user
        df['amount_zscore'] = (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1)

        # Is amount unusually high for this user (> 3 std)
        df['is_unusual_amount'] = (df['amount_zscore'].abs() > 3).astype(int)

        return df

    def get_feature_columns(self):
        """Return list of feature columns for model training"""
        return [
            # Time features
            'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',

            # Velocity features
            'txn_count_1h', 'txn_count_24h', 'amount_sum_1h', 'amount_sum_24h',

            # Amount features
            'amount', 'amount_log', 'is_round_amount', 'amount_deviation',

            # Device/location features
            'is_ios', 'is_new_device', 'is_different_city',

            # Risk scores
            'merchant_risk_score', 'amount_zscore', 'is_unusual_amount'
        ]


if __name__ == "__main__":
    # Test feature engineering
    df = pd.read_csv('../data/synthetic_upi_transactions.csv')

    engineer = FraudFeatureEngineer()
    df_features = engineer.fit_transform(df)

    print("Feature columns created:")
    print(engineer.get_feature_columns())

    print("\nSample of engineered features:")
    print(df_features[engineer.get_feature_columns()].head())

    print("\nFeature statistics:")
    print(df_features[engineer.get_feature_columns()].describe())
