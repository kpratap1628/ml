from contextlib import asynccontextmanager
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ham10000_cnn.keras"
STATS_PATH = BASE_DIR / "models" / "preprocessing.json"
INDEX_PATH = BASE_DIR / "static" / "index.html"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

CLASSES = {
    0: ("akiec", "Actinic keratoses / intraepithelial carcinoma"),
    1: ("bcc", "Basal cell carcinoma"),
    2: ("bkl", "Benign keratosis-like lesions"),
    3: ("df", "Dermatofibroma"),
    4: ("nv", "Melanocytic nevi"),
    5: ("vasc", "Vascular lesions"),
    6: ("mel", "Melanoma"),
}

DISCLAIMER = (
    "Research prototype only. Not a medical device: it cannot diagnose or rule out "
    "cancer. Consult a qualified dermatologist about any concerning lesion."
)

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists() and STATS_PATH.exists():
        # Imported here so the server starts fast even before TensorFlow is needed.
        from tensorflow.keras.models import load_model

        state["model"] = load_model(MODEL_PATH)
        state["stats"] = json.loads(STATS_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="DermaScan API", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if "model" not in state:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train it first with `python train_model.py` "
            "(needs models/ham10000_cnn.keras and models/preprocessing.json).",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a JPG or PNG file.")

    arr = np.asarray(image.resize((28, 28), Image.Resampling.LANCZOS), dtype=np.float32)
    scale = float(state["stats"].get("scale", 1.0))
    mean = float(state["stats"]["mean"])
    std = float(state["stats"]["std"]) or 1.0
    arr = (arr / scale - mean) / std

    probs = state["model"].predict(arr[np.newaxis, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    code, label = CLASSES[idx]

    return {
        "prediction": {
            "code": code,
            "label": label,
            "confidence": round(float(probs[idx]), 4),
        },
        "scores": [
            {"code": c, "label": l, "probability": round(float(probs[i]), 4)}
            for i, (c, l) in CLASSES.items()
        ],
        "disclaimer": DISCLAIMER,
    }
