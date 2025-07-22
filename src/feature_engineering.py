import pandas as pd
import numpy as np

def time_weighted_value(group):
    """Calculates a time-weighted value of transactions to emphasize recent activity."""
    group = group.sort_values('timestamp')
    time_diff = (group['timestamp'].max() - group['timestamp']).dt.days
    decay_factor = np.exp(-0.01 * time_diff) # Exponential decay
    return (group['amount_usd'] * decay_factor).sum()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers a comprehensive set of features for each wallet from transaction data.

    Args:
        df (pd.DataFrame): The preprocessed transaction DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with wallets as indices and engineered features as columns.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['amount_usd'] = pd.to_numeric(df['actionData'].apply(lambda x: x.get('amount', 0)), errors='coerce') * \
                       pd.to_numeric(df['actionData'].apply(lambda x: x.get('assetPriceUSD', 0)), errors='coerce')
    df = df.dropna(subset=['amount_usd'])

    wallets = {}
    for wallet, group in df.groupby('userWallet'):
        features = {}
        group = group.sort_values('timestamp')

        # 1. Core Financial Metrics
        deposits = group[group['action'] == 'deposit']['amount_usd'].sum()
        borrows = group[group['action'] == 'borrow']['amount_usd'].sum()
        repays = group[group['action'] == 'repay']['amount_usd'].sum()
        
        features['total_volume_usd'] = group['amount_usd'].sum()
        features['net_deposits_usd'] = deposits - (borrows - repays)
        features['ltv_ratio'] = borrows / (deposits + 1e-6)
        features['repayment_ratio'] = repays / (borrows + 1e-6)

        # 2. Behavioral & Risk Indicators
        features['liquidation_count'] = (group['action'] == 'liquidationcall').sum()
        features['unique_assets_interacted'] = group['actionData'].apply(lambda x: x.get('assetSymbol')).nunique()
        
        # 3. Temporal Features
        days_active = (group['timestamp'].max() - group['timestamp'].min()).days + 1
        features['transaction_frequency'] = len(group) / days_active
        features['time_weighted_volume'] = time_weighted_value(group)

        if borrows > 0:
            avg_borrow_amount = group[group['action'] == 'borrow']['amount_usd'].mean()
            # Time between first deposit and first borrow
            first_deposit_time = group[group['action'] == 'deposit']['timestamp'].min()
            first_borrow_time = group[group['action'] == 'borrow']['timestamp'].min()
            if pd.notna(first_deposit_time) and pd.notna(first_borrow_time):
                features['deposit_to_borrow_lag_hours'] = (first_borrow_time - first_deposit_time).total_seconds() / 3600
            else:
                features['deposit_to_borrow_lag_hours'] = -1 # Sentinel value
        else:
            avg_borrow_amount = 0
            features['deposit_to_borrow_lag_hours'] = -1

        features['avg_borrow_amount_usd'] = avg_borrow_amount
        
        # 4. Stability & Consistency
        features['transaction_value_std_dev'] = group['amount_usd'].std()
        features['is_flash_loan_user'] = 1 if group['action'].str.contains('flashLoan').any() else 0
        
        wallets[wallet] = features

    feature_df = pd.DataFrame.from_dict(wallets, orient='index').fillna(0)
    # Clamp extreme values that can skew the model
    feature_df['ltv_ratio'] = feature_df['ltv_ratio'].clip(0, 5) 
    
    return feature_df