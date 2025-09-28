import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is present (Cloud-safe)
# Use a writable cache dir inside the app's working directory
import os
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Download only what is needed (stopwords, wordnet) into local dir
nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)
nltk.download("wordnet", download_dir=NLTK_DATA_DIR, quiet=True)

# Point NLTK to our local data directory at runtime
nltk.data.path.insert(0, NLTK_DATA_DIR)

# Load model and vectorizer (keep files in repo root with this script)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
VECT_PATH = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()

# Precompute resources
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="AI Echo - Review Sentiment")
st.title("🧠 AI Echo: ChatGPT Review Sentiment Analyzer")

review_input = st.text_area("💬 Enter your ChatGPT review here:")

if st.button("🔍 Analyze Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review text.")
    else:
        cleaned = clean_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"🎯 Predicted Sentiment: **{prediction}**")
        if prediction == "Positive":
            st.balloons()
        elif prediction == "Negative":
            st.error("☹️ Seems like a bad experience!")
        else:
            st.info("😐 This feels neutral.")
