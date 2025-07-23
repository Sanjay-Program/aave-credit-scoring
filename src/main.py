import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from feature_engineering import engineer_features
from scoring_model import CreditScoringModel

def main():
    """
    Main function to run the credit scoring pipeline.
    """
    # --- IMPROVEMENT: Create robust file paths ---
    # This ensures the script can find the files regardless of where you run it from.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'user-wallet-transactions.json')
    SCORES_PATH = os.path.join(PROJECT_ROOT, 'wallet_scores.csv')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'credit_scoring_model.pkl')
    GRAPH_PATH = os.path.join(PROJECT_ROOT, 'score_distribution.png')


    # 1. Load Data
    print("Loading transaction data...")
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)

    # 2. Engineer Features
    print("Engineering features...")
    feature_df = engineer_features(df)

    # 3. Train Model and Score Wallets
    model = CreditScoringModel()
    model.train(feature_df)
    wallet_scores = model.predict(feature_df)

    # 4. Save Results
    print("Saving wallet scores...")
    wallet_scores.to_csv(SCORES_PATH)
    model.save_model(MODEL_PATH)
    print(f"Successfully scored {len(wallet_scores)} wallets.")

    # 5. Generate Analysis Graph
    print("Generating score distribution analysis...")
    plt.figure(figsize=(12, 7))
    plt.hist(wallet_scores['credit_score'], bins=range(0, 1001, 50), edgecolor='black', alpha=0.7)
    plt.title('Distribution of Wallet Credit Scores', fontsize=16)
    plt.xlabel('Credit Score Bins', fontsize=12)
    plt.ylabel('Number of Wallets', fontsize=12)
    plt.xticks(range(0, 1001, 100))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(GRAPH_PATH)
    print(f"Analysis graph saved to {GRAPH_PATH}")

if __name__ == '__main__':
    main()