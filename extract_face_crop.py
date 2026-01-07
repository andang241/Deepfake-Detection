import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(root: Path) -> List[Tuple[str, int, str]]:
    items = []
    for split in ["train", "val", "test"]:
        for cls, label in [("real", 0), ("fake", 1)]:
            d = root / split / cls
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    items.append((str(p), label, split))
    return items


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_images(paths: List[str]):
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = None
        imgs.append(img)
    return imgs


def crop_face_opencv(pil_img: Image.Image, face_cascade, margin: float = 0.35) -> Image.Image:
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return pil_img

    # chọn mặt lớn nhất
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    mx = int(w * margin)
    my = int(h * margin)

    h_img, w_img = gray.shape
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + w + mx)
    y2 = min(h_img, y + h + my)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return pil_img

    return Image.fromarray(crop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = list_images(data_root)
    if not items:
        raise SystemExit(f"No images found under: {data_root}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.eval().to(device)

    # OpenCV Haar face detector
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)
    print("[INFO] OpenCV Haar face detector ready")

    rows = []
    dim = None

    paths = [x[0] for x in items]
    labels = [x[1] for x in items]
    splits = [x[2] for x in items]

    with torch.no_grad():
        for idxs in tqdm(list(chunked(list(range(len(items))), args.batch_size)), desc="Encoding"):
            batch_paths = [paths[i] for i in idxs]
            batch_labels = [labels[i] for i in idxs]
            batch_splits = [splits[i] for i in idxs]

            pil_imgs = load_images(batch_paths)
            valid = [
                (p, y, s, im)
                for (p, y, s, im) in zip(batch_paths, batch_labels, batch_splits, pil_imgs)
                if im is not None
            ]
            if not valid:
                continue

            v_paths, v_labels, v_splits, v_imgs = zip(*valid)

            # FACE CROP
            v_imgs = [crop_face_opencv(im, face_cascade) for im in v_imgs]

            imgs_t = torch.stack([preprocess(im) for im in v_imgs], dim=0).to(device)

            feats = model.encode_image(imgs_t).float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            feats_np = feats.cpu().numpy()
            if dim is None:
                dim = feats_np.shape[1]
                print(f"[INFO] Embedding dim = {dim}")

            for p, y, s, emb in zip(v_paths, v_labels, v_splits, feats_np):
                rows.append(
                    {"path": p, "label": int(y), "split": s, "emb": emb.astype(np.float32)}
                )

    df = pd.DataFrame(rows)
    print("[INFO] Writing parquet:", out_path)
    df.to_parquet(out_path, index=False)
    print("[DONE] Saved:", out_path)
    print(df["split"].value_counts(dropna=False))
    print(df["label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
