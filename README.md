# DermaScan — HAM10000 skin lesion classifier API

FastAPI backend serving the 7-class CNN from `ham10000-skin-disease-classification.ipynb`,
plus a small upload page for testing at `/`.

## Project layout

```
app.py              FastAPI app: GET / (page), GET /health, POST /predict
train_model.py      Trains the CNN and exports models/ham10000_cnn.keras + preprocessing.json
static/index.html   Upload page
data/               Put hmnist_28_28_RGB.csv here (not committed)
models/             Trained artifacts (needed by the API)
```

## 1. Setup (Python 3.12 — TensorFlow does not support 3.14)

```bash
python3.12 -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install -r requirements.txt
```

## 2. Get the data and train

Download `hmnist_28_28_RGB.csv` from the Kaggle dataset
**skin-cancer-mnist-ham10000** and place it at `data/hmnist_28_28_RGB.csv`.

```bash
python train_model.py
```

This prints hold-out accuracy and a per-class report, then writes:

- `models/ham10000_cnn.keras` (best checkpoint by validation accuracy)
- `models/preprocessing.json` (`scale`, `mean`, `std` used for standardization)

The pipeline splits train/val/test (70/10/20) on the original data *before*
oversampling or augmentation, so validation and test metrics are honest.
Training adds flip/rotation/brightness augmentation plus Dropout/BatchNorm to
counter overfitting. Expect roughly 75–82% hold-out accuracy — far below the
notebook's ~97%, which was inflated because it oversampled before splitting.

## 3. Run locally

```bash
uvicorn app:app --reload
```

- Upload page: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

API example:

```bash
curl -F "file=@lesion.jpg" http://127.0.0.1:8000/predict
```

```json
{
  "prediction": { "code": "nv", "label": "Melanocytic nevi", "confidence": 0.97 },
  "scores": [ { "code": "akiec", "label": "...", "probability": 0.001 }, "..." ],
  "disclaimer": "..."
}
```

Classes: `akiec`, `bcc`, `bkl`, `df`, `nv`, `vasc`, `mel`.

## 4. Deploy

The model is small (under 2 MB), so committing it to Git is fine.

- **Render / Railway**: start command `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Hugging Face Spaces (Docker)**: same start command; use a Python 3.12 base image.

Requirements: `models/ham10000_cnn.keras` and `models/preprocessing.json` must be present
in the repo. Keep the dataset CSV out of Git (it is in `.gitignore`).

## Limits

- 28×28 input discards a lot of visual detail; accuracy here is not clinical-grade.
- Confidence is the model's raw softmax score, not a calibrated probability.
- Educational prototype only — not a diagnostic tool.
