# 🛒 Shopper Spectrum

## Steps to Deploy on Streamlit Cloud

1. Create a GitHub repo named `shopper-spectrum`
2. Upload ALL files from this zip into the repo root
3. Go to https://share.streamlit.io → New app → select your repo → main file: `app.py`
4. Click Deploy!

## Files in this project
- `app.py` — Main Streamlit app (4 pages)
- `clean_data.csv` — Cleaned transaction data
- `rfm_data.csv` — RFM + segment data
- `kmeans_model.pkl` — Trained KMeans model
- `scaler.pkl` — StandardScaler
- `item_sim.pkl` — Product cosine similarity matrix
- `labels_map.json` — Cluster → Segment label mapping
- `requirements.txt` — Python dependencies
