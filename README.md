# Advanced DeFi Credit Scoring Model for Aave V2

This repository contains a robust machine learning pipeline for assigning credit scores to Aave V2 wallets based on their on-chain transaction history. The model is designed to be transparent, extensible, and resistant to simplistic gaming.

## 🏛️ Method and Architecture

This project moves beyond simple weighted models and implements a more sophisticated, unsupervised learning approach to identify risky or anomalous behavior.

### 1. Feature Engineering

A wide array of features are engineered to capture the nuances of wallet behavior, focusing on financial health, risk exposure, and temporal patterns. Key features include:
* **Core Financials**: `ltv_ratio`, `repayment_ratio`, `net_deposits_usd`.
* **Risk Indicators**: `liquidation_count`, `is_flash_loan_user`.
* **Temporal Dynamics**: `transaction_frequency`, `time_weighted_volume`, and `deposit_to_borrow_lag_hours` to capture the speed and recency of actions.
* **Behavioral Stability**: Standard deviation of transaction values to measure consistency.

### 2. Processing and Modeling Pipeline

The core of the model is a `scikit-learn` pipeline designed for robustness and reproducibility:

1.  **`MinMaxScaler`**: Normalizes all features to a common scale (0-1) to prevent features with large magnitudes from dominating the model.
2.  **`PCA` (Principal Component Analysis)**: Reduces dimensionality by transforming the features into a smaller set of uncorrelated components. This step retains 95% of the variance, reducing noise and improving model performance.
3.  **`IsolationForest`**: An unsupervised anomaly detection algorithm. It works by "isolating" observations by randomly selecting a feature and then randomly selecting a split value. The logic is that anomalous points are "easier" to isolate. In our context, **risky and bot-like behaviors are treated as anomalies**.

### 3. Score Generation

The `IsolationForest` model outputs an anomaly score. This score is then transformed into an intuitive **credit score from 0 to 1000**. Wallets with normal, consistent behavior receive high scores, while anomalous, risky wallets receive low scores.

## 🚀 How to Run the Project

1.  **Setup Environment**:
    ```bash
    git clone https://github.com/Sanjay-Program/aave-credit-scoring
    cd aave-credit-scoring
    pip install -r requirements.txt
    ```

2.  **Place Data**:
    Ensure the `user-wallet-transactions.json` file is located in the `data/` directory.

3.  **Execute the Pipeline**:
    Navigate to the `src/` directory and run the main script.
    ```bash
    cd src/
    python main.py
    ```

4.  **Outputs**:
    * `wallet_scores.csv`: A CSV file in the root directory with wallet addresses and their credit scores.
    * `credit_scoring_model.pkl`: The trained and saved model pipeline in the root directory.
    * `score_distribution.png`: An analysis graph in the root directory showing the score distribution.