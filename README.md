# Car Price Predictor

Estimate the fair price of a used car in India, see how its value will fall, and check whether a seller's asking price is a good deal. Built with scikit-learn and Streamlit.

## Features

| Tab | What you get |
|---|---|
| **Estimate** | Live price estimate on a number-plate card, 80% likely range, value in 1/2/3/5 years, deal check against an asking price, depreciation curve, "what changes the price" analysis, similar real listings, EMI planner, downloadable report |
| **Compare** | Save several estimates and compare them side by side with error bars; CSV export |
| **Market explorer** | Filter the listings by brand, fuel, price and age; four interactive charts; CSV export |
| **Model insights** | Model comparison, feature importance, predicted-vs-actual chart, honest accuracy numbers |

## Project structure

```
app.py                 Streamlit app (UI only)
src/config.py          paths, feature lists, constants
src/data.py            loading + cleaning
src/estimators.py      forests constrained so older / higher-km never raises price
src/train.py           model comparison (repeated CV) and training
src/predict.py         estimate, range, what-if, deal check, EMI, similar cars
src/ui.py              CSS and chart helpers
models/                trained model.joblib + meta.json (metrics, importance)
data/car_dataset.csv   dataset
tests/                 pytest suite
.streamlit/config.toml theme
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Retrain after changing data or features: `python -m src.train`. Run tests: `pip install pytest && pytest`.

## What changed from the first version

- The original model used only 5 inputs and ignored the **car brand and model**, the biggest price driver. Cross-validated R² went from roughly 0.03 (Linear Regression) to about **0.72**.
- The target is modelled on a log scale and the model is **monotone** in age and kilometres, so predictions stay sensible.
- Data cleaning: duplicates and 3 impossible prices removed; the corrupted mileage/power/torque columns are not used.
- The original pipeline compared models by scoring on the training data. This one uses **repeated cross-validation** and a held-out test set.
- Fixed bugs: the ownership mapping lost "Fourth Owner" and had a stray trailing space that created missing values, the prediction was shown as `11 Rs.` when the price is in lakhs, and `pd.get_dummies` on a single row did not match training encodings.

## Deploy

### Streamlit Community Cloud (free, recommended)
1. Push this folder to a **GitHub repository** (`models/model.joblib` is only ~3 MB, so commit it).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick the repo, branch `main`, main file `app.py`.
3. Under **Advanced settings** choose Python 3.12, then **Deploy**. You get a public URL in a couple of minutes.

If the saved model ever fails to load (for example after a scikit-learn upgrade), the app retrains itself automatically on startup.

### Docker / Render / Hugging Face Spaces
```bash
docker build -t car-price . && docker run -p 8501:8501 car-price
```
Render and Spaces (Docker SDK) can deploy the included `Dockerfile` directly; expose port 8501.

## Limitations
About 1,100 listings after cleaning. Rare and exotic models have wide uncertainty, and condition, accident history and city are not captured. Use the estimate as a negotiation starting point, not a formal valuation.
