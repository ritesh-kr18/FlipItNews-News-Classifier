# FlipItNews — News Category Classifier

A Streamlit app that classifies a news headline or article into one of five BBC News categories: business, entertainment, politics, sport, or tech.

## Models

- **TF-IDF + feed-forward neural network** (`tfidf_vectorizer.joblib` + `tfidf_nn.joblib`): recommended default because it performed best in the notebook's comparison.
- **PyTorch bidirectional GRU** (`flipitnews_rnn.pt`): shown as an option when its checkpoint and configuration files are present.

The app reads the saved artifacts produced by the Lab 9 notebook. It does not retrain models during deployment.

## 1. Required project structure

After running the notebook's final save-artifacts cell, copy its complete `lab9_artifacts` folder beside `app.py` so the project looks like this:

```text
FlipItNews/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── lab9_artifacts/
    ├── label_classes.json
    ├── tfidf_vectorizer.joblib
    ├── tfidf_nn.joblib
    ├── model_metrics.csv
    ├── flipitnews_rnn.pt
    ├── vocab.json
    ├── rnn_config.json
    └── word2vec.model              # saved by notebook; app does not need it
```

The app requires the first four files. The PyTorch GRU option additionally needs `flipitnews_rnn.pt`, `vocab.json`, and `rnn_config.json`. `word2vec.model` is not required by this dashboard.

**Important:** The model artifacts are generated on your computer when you run the notebook. They are not included in this starter package. Copy the real files from the notebook's `lab9_artifacts/` directory; do not create empty placeholder files.

## 2. Run locally 

Open Terminal and move into this project folder:

```bash
cd path/to/FlipItNews
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

