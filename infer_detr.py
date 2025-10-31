# infer_detr.py
import os, glob, json, csv, torch, cv2
from tqdm import tqdm
from PIL import Image
import transformers
from transformers import DetrImageProcessor, DetrForObjectDetection

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _list_images(folder):
    paths = []
    for p in glob.glob(os.path.join(folder, "*")):
        if os.path.isdir(p):
            # include nested subfolders if any
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(IMG_EXTS):
                        paths.append(os.path.join(root, f))
        elif p.lower().endswith(IMG_EXTS):
            paths.append(p)
    return sorted(paths)

def _load_processor(pref):
    try:
        return DetrImageProcessor.from_pretrained(pref)
    except Exception:
        return DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def _load_model(checkpoint_path):
    # If a dir with HF weights
    if os.path.isdir(checkpoint_path):
        model = DetrForObjectDetection.from_pretrained(checkpoint_path)
    else:
        # Lightning .ckpt
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", ignore_mismatched_sizes=True
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        sd = state.get("state_dict", state)
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    return model

def infer_folder(
    images_dir,
    output_dir,
    checkpoint_path,                 # HF dir or Lightning .ckpt
    conf_thresh=0.5,
    device=None
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load processor (prefer same dir as model if available)
    processor = _load_processor(checkpoint_path)
    model = _load_model(checkpoint_path)
    model.to(device).eval()

    # labels
    id2label = getattr(model.config, "id2label", None) or {}
    # optional override from a sidecar file (if you saved it)
    sidecar = os.path.join(checkpoint_path, "id2label.json") if os.path.isdir(checkpoint_path) else None
    if sidecar and os.path.exists(sidecar):
        with open(sidecar) as f:
            id2label = {int(k): v for k, v in json.load(f).items()}

    img_paths = _list_images(images_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    csv_path = os.path.join(output_dir, "detections.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["filename", "cls_id", "cls_name", "score", "x_min", "y_min", "x_max", "y_max"])

        for p in tqdm(img_paths, desc="Inferring"):
            image = Image.open(p).convert("RGB")
            enc = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model(**enc)

            results = processor.post_process_object_detection(
                out,
                target_sizes=torch.tensor([image.size[::-1]], device=device),
                threshold=conf_thresh
            )[0]

            # draw using PIL->NumPy to avoid cv2.imread path issues
            img_np = np.array(image)  # RGB
            img_cv = img_np.copy()

            per_image_json = []
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                x0, y0, x1, y1 = [int(v.item()) for v in box]
                cls_id = int(label.item())
                cls_name = id2label.get(cls_id, str(cls_id))
                sc = float(score.item())

                writer.writerow([os.path.basename(p), cls_id, cls_name, f"{sc:.4f}", x0, y0, x1, y1])

                per_image_json.append({
                    "filename": os.path.basename(p),
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "score": sc,
                    "bbox_xyxy": [x0, y0, x1, y1]
                })

                cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 255, 0), 2)
                text = f"{cls_name}:{sc:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_text = max(0, y0 - 6)
                cv2.rectangle(img_cv, (x0, y_text - th - 4), (x0 + tw + 4, y_text + 2), (0, 255, 0), -1)
                cv2.putText(img_cv, text, (x0 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # save annotated image
            det_name = os.path.splitext(os.path.basename(p))[0] + "_det.png"
            det_path = os.path.join(output_dir, "images", det_name)
            cv2.imwrite(det_path, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))

            # save per-image JSON
            json_path = os.path.join(output_dir, "labels", os.path.splitext(os.path.basename(p))[0] + ".json")
            with open(json_path, "w") as fj:
                json.dump(per_image_json, fj)

    print(f"Saved images → {os.path.join(output_dir,'images')}")
    print(f"Saved JSONs → {os.path.join(output_dir,'labels')}")
    print(f"Saved CSV   → {csv_path}")