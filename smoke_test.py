"""Smoke test: send real (held-out) HAM10000 samples through the running API."""
import io
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split

API = "http://127.0.0.1:8000"
CODES = {i: c for i, c in enumerate(["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"])}

df = pd.read_csv("data/hmnist_28_28_RGB.csv")
y = df["label"].astype("int64").to_numpy()
x = df.drop(columns=["label"]).to_numpy(dtype=np.uint8).reshape(-1, 28, 28, 3)
_, X_rest, _, y_rest = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
_, X_test, _, y_test = train_test_split(X_rest, y_rest, test_size=2 / 3, random_state=1, stratify=y_rest)

rng = np.random.default_rng(0)
idx = rng.choice(len(X_test), 10, replace=False)

correct = 0
for i in idx:
    img = Image.fromarray(X_test[i]).resize((224, 224), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    res = requests.post(f"{API}/predict", files={"file": ("sample.png", buf, "image/png")})
    data = res.json()
    pred = data["prediction"]["code"]
    truth = CODES[int(y_test[i])]
    ok = pred == truth
    correct += ok
    print(f"true={truth:5s} pred={pred:5s} conf={data['prediction']['confidence']:.2f} {'OK' if ok else 'X'}")

print(f"\n{correct}/{len(idx)} correct on held-out samples")
