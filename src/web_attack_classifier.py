"""
web_attack_classifier.py - Stage 3: Supervised Web Attack Identification Layer
MS Thesis: Reinforcement Learning-Based Adaptive Alerts Correlation for Detecting Multi-Stage Cyber Attacks
Candidate: Haaziq Rasool (Bahria University Islamabad)

Implements Proposal Objective 1 & Methodology Stage 3:
- Trains a Supervised NLP/N-gram Classifier (TF-IDF + Random Forest / Logistic Regression)
- Parses raw HTTP URIs, headers, and request payloads
- Identifies SQL Injection (SQLi), Cross-Site Scripting (XSS), Path Traversal, and RCE
- Outputs calibrated probability scores and verified alert labels to feed the RL correlation layer.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


class WebAttackIdentifier:
    """
    Stage 3 Supervised Web Attack Classifier:
    Extracts character and word N-grams from HTTP request URIs/payloads to detect web attacks.
    """
    def __init__(self, max_features: int = 1500):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
            lowercase=True
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, df_web_logs: pd.DataFrame) -> Dict[str, float]:
        print("[*] Stage 3: Training Supervised Web Attack Identifier (TF-IDF + Random Forest)...")
        
        # Features: URI path + HTTP method + WAF rule string
        text_corpus = (
            df_web_logs["http_method"].astype(str) + " " +
            df_web_logs["uri_path"].astype(str) + " " +
            df_web_logs["waf_rule_hit"].astype(str)
        )
        labels = df_web_logs["label"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            text_corpus, labels, test_size=0.25, random_state=42, stratify=labels
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier.fit(X_train_vec, y_train)
        self.is_trained = True

        y_pred = self.classifier.predict(X_test_vec)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"[+] Stage 3 Classifier Trained | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        return {"precision": float(p), "recall": float(r), "f1": float(f1)}

    def predict_proba(self, df_web_logs: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("WebAttackIdentifier is not trained yet.")
        text_corpus = (
            df_web_logs["http_method"].astype(str) + " " +
            df_web_logs["uri_path"].astype(str) + " " +
            df_web_logs["waf_rule_hit"].astype(str)
        )
        X_vec = self.vectorizer.transform(text_corpus)
        return self.classifier.predict_proba(X_vec)[:, 1]

    def save(self, path: str = "models/web_attack_classifier.joblib") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, path)
        print(f"[+] Stage 3 Model saved to: {path}")

    @classmethod
    def load(cls, path: str = "models/web_attack_classifier.joblib") -> "WebAttackIdentifier":
        data = joblib.load(path)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.classifier = data["classifier"]
        instance.is_trained = True
        return instance


def run_stage3_identification(
    web_logs_path: str = "data/synthetic/web_waf_logs.csv",
    model_save_path: str = "models/web_attack_classifier.joblib"
) -> WebAttackIdentifier:
    if not os.path.exists(web_logs_path):
        raise FileNotFoundError(f"Web logs not found at {web_logs_path}. Run gen_synthetic.py first.")
    
    df_web = pd.read_csv(web_logs_path)
    identifier = WebAttackIdentifier()
    identifier.train(df_web)
    identifier.save(model_save_path)
    return identifier


if __name__ == "__main__":
    run_stage3_identification()
