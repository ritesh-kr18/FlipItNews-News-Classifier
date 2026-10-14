import json
import re
import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "lab9_artifacts"
DEVICE = torch.device("cpu")  # Streamlit hosting commonly uses CPU

st.set_page_config(page_title="FlipItNews | News Classifier", page_icon="📰", layout="wide")


@st.cache_resource
def load_artifacts():
    required = [
        ARTIFACT_DIR / "label_classes.json",
        ARTIFACT_DIR / "tfidf_vectorizer.joblib",
        ARTIFACT_DIR / "tfidf_nn.joblib",
        ARTIFACT_DIR / "model_metrics.csv",
    ]
    missing = [str(p.relative_to(BASE_DIR)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing deployment artifacts: " + ", ".join(missing) +
            ". Copy the lab9_artifacts folder saved by the notebook into the project root."
        )

    with open(ARTIFACT_DIR / "label_classes.json", encoding="utf-8") as f:
        classes = json.load(f)
    vectorizer = joblib.load(ARTIFACT_DIR / "tfidf_vectorizer.joblib")
    tfidf_model = joblib.load(ARTIFACT_DIR / "tfidf_nn.joblib")
    metrics = pd.read_csv(ARTIFACT_DIR / "model_metrics.csv")

    gru_model = None
    gru_vocab = None
    gru_config = None
    gru_path = ARTIFACT_DIR / "flipitnews_rnn.pt"
    vocab_path = ARTIFACT_DIR / "vocab.json"
    config_path = ARTIFACT_DIR / "rnn_config.json"
    if gru_path.exists() and vocab_path.exists() and config_path.exists():
        with open(vocab_path, encoding="utf-8") as f:
            gru_vocab = json.load(f)
        with open(config_path, encoding="utf-8") as f:
            gru_config = json.load(f)
        gru_model = NewsRNN(
            gru_config["vocab_size"], gru_config["num_classes"],
            gru_config["embedding_dim"], gru_config["hidden_size"]
        ).to(DEVICE)
        state = torch.load(gru_path, map_location=DEVICE, weights_only=True)
        gru_model.load_state_dict(state)
        gru_model.eval()
    return classes, vectorizer, tfidf_model, metrics, gru_model, gru_vocab, gru_config


class NewsRNN(nn.Module):
    """Must match the bidirectional GRU architecture used in the notebook."""
    def __init__(self, vocab_size, num_classes, embedding_dim=64, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size,
            batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, tokens, lengths):
        from torch.nn.utils.rnn import pack_padded_sequence
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(features))


try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    try:
        STOP_WORDS = set(nltk_stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(nltk_stopwords.words("english"))
except Exception:
    STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    """Match the notebook's lowercase/URL/HTML/punctuation/stopword preprocessing."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"[a-z]+", text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def encode_rnn(text, vocab, max_len):
    tokens = preprocess_text(text).split()[:max_len]
    ids = [vocab.get(token, 1) for token in tokens]
    length = max(1, len(ids))
    ids += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long), torch.tensor([length], dtype=torch.long)


def predict_tfidf(text, vectorizer, model, classes):
    clean = preprocess_text(text)
    features = vectorizer.transform([clean])
    pred_id = int(model.predict(features)[0])
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
    return classes[pred_id], probs


def predict_rnn(text, model, vocab, config, classes):
    tokens, lengths = encode_rnn(text, vocab, int(config["max_len"]))
    model.eval()
    with torch.no_grad():
        logits = model(tokens.to(DEVICE), lengths.to(DEVICE))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs))
    return classes[pred_id], probs


st.title("📰 FlipItNews")
st.subheader("News category classification")
st.write("Classify a news headline or article into **business, entertainment, politics, sport, or tech**.")

try:
    classes, vectorizer, tfidf_model, metrics, rnn_model, rnn_vocab, rnn_config = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info("Expected structure: app.py and a lab9_artifacts/ folder in the same project directory.")
    st.stop()

with st.sidebar:
    st.header("About the project")
    st.write("Models: TF-IDF + feed-forward neural network, and optional PyTorch bidirectional GRU.")
    st.write("Dataset: BBC News articles (five categories).")
    st.caption("Predictions are model estimates, not editorial verification.")

st.markdown("### Model performance from your notebook")
if not metrics.empty:
    st.dataframe(metrics.style.format({c: "{:.4f}" for c in metrics.select_dtypes(include="number").columns}), use_container_width=True, hide_index=True)

st.markdown("### Classify an article")
model_choice = st.radio(
    "Choose model",
    ["TF-IDF Neural Network"] + (["PyTorch BiGRU (RNN)"] if rnn_model is not None else []),
    horizontal=True,
)
article = st.text_area(
    "Paste a headline or news article",
    height=220,
    placeholder="Example: The government announced new economic measures...",
)

if st.button("Classify news", type="primary"):
    if not article.strip():
        st.warning("Please enter a headline or article first.")
    else:
        with st.spinner("Classifying article..."):
            if model_choice == "TF-IDF Neural Network":
                category, probs = predict_tfidf(article, vectorizer, tfidf_model, classes)
            else:
                category, probs = predict_rnn(article, rnn_model, rnn_vocab, rnn_config, classes)
        st.success(f"Predicted category: **{category.upper()}**")
        if probs is not None:
            prob_df = pd.DataFrame({"Category": classes, "Score": np.asarray(probs, dtype=float)})
            prob_df = prob_df.sort_values("Score", ascending=False)
            st.markdown("#### Class scores")
            st.bar_chart(prob_df.set_index("Category"))
            st.dataframe(prob_df.assign(Score=prob_df["Score"].map(lambda x: f"{x:.2%}")), hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("Built with Python, scikit-learn, PyTorch, and Streamlit.")
