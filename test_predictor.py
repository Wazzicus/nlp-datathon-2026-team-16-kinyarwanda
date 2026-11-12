
from predictor import SentimentPredictor

predictor = SentimentPredictor()

examples = [
    "Nishimiye cyane uko iki gikorwa cyagenze.",
    "Serivisi twahawe yari mbi cyane.",
    "Inama izaba ku wa mbere saa tatu za mu gitondo.",
    "Ntabwo nishimiye uko ikibazo cyakemuwe."
]

for text in examples:
    result = predictor.predict(text)

    print("\nText:", result["text"])
    print("Prediction:", result["sentiment"])
    print("Decision scores:", result["decision_scores"])
