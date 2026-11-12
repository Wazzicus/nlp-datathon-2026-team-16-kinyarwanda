
from pathlib import Path
import joblib


class SentimentPredictor:
    def __init__(self, model_path="artifacts/baseline/model.joblib"):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.pack = joblib.load(model_path)

        if self.pack.get("track") != "sentiment":
            raise ValueError("Loaded model is not a sentiment model.")

        if self.pack.get("practice") is not False:
            raise ValueError("Practice models cannot be used for final inference.")

        self.model = self.pack["model"]

        self.labels = ["positive", "negative", "neutral"]

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please provide non-empty text.")

        prediction = self.model.predict([text])[0]
        decision_scores = self.model.decision_function([text])[0]

        scores = {
            label: float(score)
            for label, score in zip(self.model.classes_, decision_scores)
        }

        return {
            "text": text,
            "sentiment": prediction,
            "decision_scores": scores
        }
