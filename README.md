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

## 2. Run locally on macOS

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

## 3. Upload to GitHub

1. Create a **new empty repository** on GitHub, for example `FlipItNews-News-Classifier`. Do not initialize it with a README if you already have one locally.
2. Make sure `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and the real `lab9_artifacts/` folder are inside the project directory.
3. In Terminal, from that directory, run:

```bash
git init
git add app.py requirements.txt README.md .gitignore lab9_artifacts
git status
git commit -m "Add FlipItNews news classification app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/FlipItNews-News-Classifier.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If Git says a remote named `origin` already exists, use `git remote set-url origin https://github.com/YOUR_USERNAME/FlipItNews-News-Classifier.git` instead of `git remote add origin ...`.

Before committing, inspect `git status` and ensure the model files are present. GitHub blocks individual files larger than 100 MB. If any saved artifact is that large, use Git LFS or host the artifact separately and modify the app to download it securely. Do not commit secrets or credentials.

## 4. Deploy on Streamlit Community Cloud

1. Push the project to GitHub using the instructions above.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app** (or **Deploy an app**).
4. Select the repository, branch `main`, and main file path `app.py`.
5. Deploy. Streamlit Cloud installs packages from `requirements.txt` and runs the app.
6. Open the deployed URL and test both model options, if the GRU artifacts were included.

If deployment fails, open the app's **Manage app → Logs** and read the first error. Common causes are missing `lab9_artifacts` files, an incompatible package version, or an artifact too large for GitHub.

## 5. Keep the app consistent with the notebook

- Run the notebook through the final save-artifacts cell after any retraining.
- Copy the newly generated `lab9_artifacts/` folder into this project and push the updated files.
- Do not use a checkpoint from a different model architecture with this app.
- The app's text preprocessing should match the preprocessing used during training. If you change preprocessing in the notebook, update `preprocess_text()` here too.

## Files saved by the notebook

The notebook saves `flipitnews_rnn.pt`, `vocab.json`, `label_classes.json`, `rnn_config.json`, `word2vec.model`, `tfidf_vectorizer.joblib`, `tfidf_nn.joblib`, and `model_metrics.csv` inside `lab9_artifacts/`.
