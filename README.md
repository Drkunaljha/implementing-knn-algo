# ID3 Decision Tree Streamlit App

This small project contains a Streamlit app that demonstrates two algorithms:

- ID3 decision tree (categorical attributes)
- K-Nearest Neighbors (KNN) using scikit-learn (numeric and categorical via one-hot encoding)

Files added:
- `streamlit_app.py` — main Streamlit app (use `streamlit run streamlit_app.py`).
- `play_tennis.csv` — sample dataset (used by default).
- `requirements.txt` — Python dependencies.
- `.gitignore` — common ignores.

How to run locally

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app locally:

```powershell
streamlit run streamlit_app.py
```

Deploying to GitHub and Streamlit Community Cloud

I can't create a remote GitHub repo or connect to Streamlit on your behalf without your credentials. Below are the exact steps to publish and deploy from your machine.

1. Create a GitHub repository (via website or GitHub CLI). From your project folder run:

```powershell
git init
git add .
git commit -m "Initial commit: Streamlit ID3 app"
git branch -M main
# Replace the URL below with the repo URL you create on GitHub
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

2. On Streamlit Community Cloud (https://share.streamlit.io) click "New app", connect your GitHub account, pick the repo and the `streamlit_app.py` file as the app entrypoint, and deploy.

If you'd like, I can generate the Git commands and walk you step-by-step or help push if you provide a personal access token or run the commands on your machine while I guide you.
