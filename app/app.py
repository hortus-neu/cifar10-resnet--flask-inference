# --- import -- #
from __future__ import annotations
import io 
from pathlib import Path
from typing import List

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

# --- Paths --- #
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT/"weights"/"resnet18_finetune_best.pt"

# --- Config --- #
DEVICE = torch.device("cpu")
NUM_CLASSES  = 10
CLASS_NAMES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def build_model(weights_path: Path) -> nn.Module:
    model = resnet18(weights=None)               
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, NUM_CLASSES)

    state = torch.load(str(weights_path), map_location=DEVICE)

    model.load_state_dict(state, strict=True)

    model.to(DEVICE)
    model.eval()
    return model

# --- APP --- #
app = Flask(__name__)
MODEL = build_model(WEIGHTS_PATH)

# --- Paths --- #
@app.get("/health")
def health():
    return jsonify({
        "status" : "ok" if WEIGHTS_PATH.exists() else "missing weights",
        "device" : str(DEVICE),
        "weights" : str(WEIGHTS_PATH)
    })
    
@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify(error="Missing file field 'image'"), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400
    
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as e:
        return jsonify(error=f"Failed to read image: {e}"), 400
    
    with torch.no_grad():
        x = transform(img).unsqueeze(0).to(DEVICE) # [1,3,32,32]
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0) # [10]
        top3_prob, top3_idx = torch.topk(probs, k=3)
    
    top1_idx = int(top3_idx[0])
        
    result = {
        "top1": {
            "label": CLASS_NAMES[top1_idx],
            "prob": float(top3_prob[0]),
            "index": top1_idx
        },
        "top3": [
            {"label": CLASS_NAMES[int(i)], 
            "prob": float(p), 
            "index": int(i)}
            for i, p in zip(top3_idx.tolist(), top3_prob.tolist())
        ]
    }
    return jsonify(result)

@app.get("/")
def index():
    return (
        "<h3>CIFAR-10 Inference API</h3>"
        "<form method='POST' action='/predict' enctype='multipart/form-data'>"
        "<input type='file' name='image' accept='image/*'/>"
        "<button type='submit'>Predict</button>"
        "</form>"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)     