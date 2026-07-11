"""
model.py — Two models for match prediction.

1. OutcomeModel  (XGBoost classifier)
   → Predicts P(away win), P(draw), P(home win)

2. GoalsModel    (Poisson regression via sklearn)
   → Predicts expected goals for home and away team separately
   → Sample from Poisson distribution to simulate scores

Train/test split is TIME-BASED (not random) to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from config import TRAIN_END, TEST_START, RANDOM_STATE
from features import FEATURE_COLS


class OutcomeModel:
    """RandomForest classifier: predicts match outcome (0=A, 1=D, 2=H)."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def fit(self, feat: pd.DataFrame):
        train = feat[feat["date"] <= TRAIN_END]
        X = train[FEATURE_COLS]
        y = train["target_result"]
        self.model.fit(X, y)
        print(f"[outcome] Trained on {len(train):,} matches (up to {TRAIN_END})")
        return self

    def evaluate(self, feat: pd.DataFrame):
        test = feat[(feat["date"] >= TEST_START)]
        if len(test) == 0:
            print("[outcome] No test data available")
            return
        X = test[FEATURE_COLS]
        y = test["target_result"]
        preds = self.model.predict(X)
        proba = self.model.predict_proba(X)
        acc  = accuracy_score(y, preds)
        loss = log_loss(y, proba)
        print(f"[outcome] Test accuracy: {acc:.3f}  |  Log-loss: {loss:.3f}  ({len(test):,} matches)")
        return acc, loss

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return [P(away), P(draw), P(home)] for each row."""
        return self.model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=FEATURE_COLS
        ).sort_values(ascending=False)


class GoalsModel:
    """
    Separate Poisson regressors for home goals and away goals.
    Poisson is the industry-standard distribution for football scoring.
    """

    def __init__(self):
        self.home_model = PoissonRegressor(alpha=0.1, max_iter=300)
        self.away_model = PoissonRegressor(alpha=0.1, max_iter=300)
        self.scaler = StandardScaler()

    def fit(self, feat: pd.DataFrame):
        train = feat[feat["date"] <= TRAIN_END]
        X_raw = train[FEATURE_COLS]
        X = self.scaler.fit_transform(X_raw)
        self.home_model.fit(X, train["target_home_goals"])
        self.away_model.fit(X, train["target_away_goals"])
        print(f"[goals]   Poisson models trained on {len(train):,} matches")
        return self

    def predict_lambda(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lambda_home, lambda_away) — expected goals per team."""
        Xs = self.scaler.transform(X)
        lam_h = self.home_model.predict(Xs)
        lam_a = self.away_model.predict(Xs)
        # Clamp to reasonable range (0.1 – 6 goals)
        lam_h = np.clip(lam_h, 0.1, 6.0)
        lam_a = np.clip(lam_a, 0.1, 6.0)
        return lam_h, lam_a

    def simulate_score(self, X: pd.DataFrame, n: int = 1) -> tuple:
        """
        Draw n scorelines from Poisson distributions.
        Returns (home_goals_array, away_goals_array) of shape (n, len(X)).
        """
        lam_h, lam_a = self.predict_lambda(X)
        home_goals = np.random.poisson(lam_h, size=(n, len(lam_h)))
        away_goals = np.random.poisson(lam_a, size=(n, len(lam_a)))
        return home_goals, away_goals


def train_all(feat: pd.DataFrame):
    """Train both models and return them."""
    outcome = OutcomeModel().fit(feat)
    goals   = GoalsModel().fit(feat)
    outcome.evaluate(feat)
    return outcome, goals
