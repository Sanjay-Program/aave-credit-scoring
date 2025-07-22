import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib

class CreditScoringModel:
    """
    A class to train and use an Isolation Forest-based credit scoring model.
    Anomalous (potentially risky) wallets are given lower scores.
    """
    def __init__(self):
        # Pipeline for preprocessing and modeling
        self.pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('pca', PCA(n_components=0.95)), # Retain 95% of variance
            ('model', IsolationForest(contamination='auto', random_state=42))
        ])

    def train(self, X: pd.DataFrame):
        """
        Trains the Isolation Forest model on the feature set.
        
        Args:
            X (pd.DataFrame): The engineered features for all wallets.
        """
        print("Training the credit scoring model...")
        self.pipeline.fit(X)
        print("Model training complete.")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts anomaly scores and converts them into a 0-1000 credit score.
        
        Args:
            X (pd.DataFrame): The engineered features for wallets to be scored.
            
        Returns:
            pd.DataFrame: A DataFrame containing the credit scores.
        """
        # score_samples returns the opposite of the anomaly score. Higher is better.
        anomaly_scores = self.pipeline.score_samples(X)
        
        # Normalize these scores to a 0-1 range
        scaler = MinMaxScaler()
        normalized_scores = scaler.fit_transform(anomaly_scores.reshape(-1, 1))
        
        # Scale to 0-1000 and invert (so high anomaly score = low credit score)
        credit_scores = (1 - normalized_scores) * 1000
        
        score_df = pd.DataFrame(credit_scores, index=X.index, columns=['credit_score'])
        return score_df.astype(int)

    def save_model(self, path: str):
        """Saves the trained pipeline to a file."""
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path: str):
        """Loads a trained pipeline from a file."""
        pipeline = joblib.load(path)
        model = CreditScoringModel()
        model.pipeline = pipeline
        print(f"Model loaded from {path}")
        return model