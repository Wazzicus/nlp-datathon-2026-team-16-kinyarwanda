
import streamlit as st
from predictor import SentimentPredictor

st.set_page_config(
    page_title="Kinyarwanda Sentiment Analysis",
    page_icon="🇷🇼",
    layout="centered"
)

@st.cache_resource
def load_predictor():
    return SentimentPredictor()

predictor = load_predictor()

st.title("Senti-RW")
st.caption("An NLP system for classifying Kinyarwanda text.")

st.markdown("""
Enter a Kinyarwanda sentence below to analyse its sentiment.
The model classifies text into three categories.
""")

examples = {
    "Choose an example...": "",
    "Positive example": "Nishimiye cyane uko iki gikorwa cyagenze.",
    "Negative example": "Serivisi twahawe yari mbi cyane.",
    "Neutral example": "Inama izaba ku wa mbere saa tatu za mu gitondo."
}

selected_example = st.selectbox(
    "Or try an example",
    list(examples.keys())
)

text = st.text_area(
    "Enter Kinyarwanda text",
    value=examples[selected_example],
    placeholder="Andika interuro mu Kinyarwanda...",
    height=150
)

if st.button("Analyse sentiment", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            result = predictor.predict(text)

            sentiment = result["sentiment"]

            sentiment_display = {
                "positive": ("Positive"),
                "negative": ("Negative"),
                "neutral": ("Neutral")
            }

            label = sentiment_display[sentiment]

            st.divider()
            st.subheader("Analysis result")
            st.markdown(f"## {label}")

            st.caption(
                "This is the model's predicted sentiment, "
                "not a guaranteed interpretation."
            )

            with st.expander("View model decision scores"):
                st.write(
                    "These are raw LinearSVC decision scores, "
                    "not probabilities."
                )

                st.json(result["decision_scores"])

        except Exception as error:
            st.error(f"An error occurred: {error}")

st.divider()

st.markdown("""
**About this project**

This demonstration uses a character-level TF-IDF vectorizer
and a Linear Support Vector Classifier (LinearSVC).

The model was trained using an original Kinyarwanda
sentiment dataset.

**Classes:** Positive · Negative · Neutral

*Research prototype — predictions may be incorrect.
""")
