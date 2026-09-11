"""
Local inference for the EfficientNet-B6 diabetic retinopathy model.

Reproduces the EXACT preprocessing and architecture from efficient-net-b6.ipynb.
If any of it drifts, predictions silently go wrong -- so nothing here is "cleaned up".

    pip install torch torchvision timm opencv-python numpy
    # (onnxruntime-gpu instead of torch/timm if you use the ONNX path)

Usage:
    python dr_inference.py retina.png
    python dr_inference.py folder_of_images/

    from dr_inference import DRPredictor
    p = DRPredictor("best_model_kappa_512.pth")
    p.predict("retina.png")
"""

import os
import sys
import json
import glob

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIG -- edit these two
# ----------------------------------------------------------------------------

MODEL_PATH = "dr_model.onnx"

# The notebook printed these during "Step 4: Optimizing rounding thresholds":
#     Optimized thresholds: [0.5xx 1.4xx 2.3xx 3.6xx]
# Copy those four numbers here. If you lost the log, leave the defaults --
# you lose roughly 0.01-0.02 kappa but nothing breaks.
THRESHOLDS = [0.5, 1.5, 2.5, 3.5]

IMAGE_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


# ----------------------------------------------------------------------------
# PREPROCESSING -- copied verbatim from the notebook
# ----------------------------------------------------------------------------

def circle_crop(image):
    """Crop to the bounding box of the fundus circle. Expects BGR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y + h, x:x + w]


def apply_ben_graham_preprocessing(image, sigmaX=30):
    """Subtract local average colour. This is what makes the lesions pop."""
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX)
    return cv2.addWeighted(image, 4, blurred_image, -4, 128)


def preprocess(src, image_size=IMAGE_SIZE):
    """
    Path / bytes / BGR array  ->  normalized CHW float32 tensor-ready array.

    Order matters and matches the notebook exactly:
      imread (BGR) -> circle_crop -> resize -> ben graham -> BGR2RGB -> normalize
    """
    if isinstance(src, (str, os.PathLike)):
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read image: {src}")
    elif isinstance(src, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image bytes")
    else:
        img = np.asarray(src)  # assumed BGR

    img = circle_crop(img)
    img = cv2.resize(img, (image_size, image_size))
    img = apply_ben_graham_preprocessing(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


# ----------------------------------------------------------------------------
# MODEL -- architecture copied verbatim so the state_dict keys line up
# ----------------------------------------------------------------------------

def _build_torch_model(model_name="efficientnet_b6"):
    import torch.nn as nn
    import timm

    def replace_batchnorm_with_groupnorm(module, num_groups=32):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                if num_channels % num_groups == 0:
                    setattr(module, name, nn.GroupNorm(num_groups=num_groups,
                                                       num_channels=num_channels))
            else:
                replace_batchnorm_with_groupnorm(child, num_groups)

    class EfficientNetModel(nn.Module):
        def __init__(self, model_name):
            super().__init__()
            self.model = timm.create_model(model_name, pretrained=False)
            replace_batchnorm_with_groupnorm(self.model)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(nn.Linear(in_features, 1))

        def forward(self, x):
            return self.model(x)

    return EfficientNetModel(model_name)


# ----------------------------------------------------------------------------
# PREDICTOR
# ----------------------------------------------------------------------------

class DRPredictor:
    """
    Works with either the .pth (PyTorch) or the .onnx export.
    Load once at app startup, then call .predict() per image.
    """

    def __init__(self, model_path=MODEL_PATH, thresholds=None, device=None,
                 image_size=IMAGE_SIZE, model_name="efficientnet_b6"):
        self.model_path = model_path
        self.image_size = image_size
        self.thresholds = np.sort(np.array(thresholds if thresholds is not None
                                           else THRESHOLDS, dtype=np.float32))
        self.backend = "onnx" if str(model_path).endswith(".onnx") else "torch"

        if self.backend == "onnx":
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"[onnx] loaded {model_path} | providers: {self.session.get_providers()}")
        else:
            import torch
            self.torch = torch
            self.device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = _build_torch_model(model_name)

            state = torch.load(model_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[warn] missing keys: {len(missing)}, unexpected: {len(unexpected)}")
                if len(missing) > 10:
                    print("       -> this usually means the .bin you loaded is the "
                            "PRETRAINED backbone, not the trained DR model")
                    print("       first 5 missing:", missing[:5])
            self.model.to(self.device).eval()
            print(f"[torch] loaded {model_path} on {self.device}")

    def _forward(self, batch):
        """batch: (N, 3, H, W) float32 -> (N,) float32"""
        if self.backend == "onnx":
            out = self.session.run(None, {self.input_name: batch})[0]
            return np.asarray(out).reshape(-1)
        with self.torch.no_grad():
            x = self.torch.from_numpy(batch).to(self.device)
            out = self.model(x)
        return out.detach().cpu().numpy().reshape(-1)

    def predict(self, src, tta_hflip=True):
        """Returns a dict with the grade, label, raw score and confidence."""
        arr = preprocess(src, self.image_size)
        batch = arr[None, ...]
        if tta_hflip:
            batch = np.concatenate([batch, arr[:, :, ::-1][None, ...].copy()], axis=0)

        preds = self._forward(batch.astype(np.float32))
        score = float(np.mean(preds))
        grade = int(np.digitize(score, self.thresholds))
        confidence = float(1.0 - min(abs(score - grade), 1.0))

        return {
            "grade": grade,
            "label": LABELS[grade],
            "raw_score": round(score, 4),
            "confidence": round(confidence, 3),
            "per_model": preds.tolist(),
        }

    def predict_batch(self, sources, batch_size=4, tta_hflip=False):
        """For a folder of images. Returns a list of dicts in the same order."""
        results = []
        for i in range(0, len(sources), batch_size):
            chunk = sources[i:i + batch_size]
            batch = np.stack([preprocess(s, self.image_size) for s in chunk]).astype(np.float32)
            if tta_hflip:
                flipped = batch[:, :, :, ::-1].copy()
                scores = (self._forward(batch) + self._forward(flipped)) / 2
            else:
                scores = self._forward(batch)
            for src, score in zip(chunk, scores):
                grade = int(np.digitize(float(score), self.thresholds))
                results.append({
                    "file": os.path.basename(str(src)),
                    "grade": grade,
                    "label": LABELS[grade],
                    "raw_score": round(float(score), 4),
                })
        return results


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH

    if not os.path.exists(model_path):
        print(f"model not found: {model_path}")
        sys.exit(1)

    predictor = DRPredictor(model_path)

    if os.path.isdir(target):
        files = sorted(
            f for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG")
            for f in glob.glob(os.path.join(target, f"*.{ext}"))
        )
        print(f"\n{len(files)} images\n")
        for r in predictor.predict_batch(files, tta_hflip=True):
            print(f"{r['file']:<30} grade {r['grade']}  {r['label']:<18} "
                  f"(raw {r['raw_score']})")
    else:
        print(json.dumps(predictor.predict(target), indent=2))
