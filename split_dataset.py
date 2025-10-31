import os
import json
import random
import shutil

# ---------------- CONFIGURATION ----------------
IMAGES_DIR = "project_dataset/images"          # folder containing all images + result.json
ANNOTATION_FILE = "result.json"           # COCO format JSON
OUTPUT_DIR = "dataset_split"     # destination for split folders

TRAIN_SPLIT = 0.7
VAL_SPLIT   = 0.2
TEST_SPLIT  = 0.1

# ------------------------------------------------


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_dataset():
    # Find annotation file (in same folder or parent)
    ann_path_same = os.path.join(IMAGES_DIR, ANNOTATION_FILE)
    ann_path_parent = os.path.join(os.path.dirname(IMAGES_DIR), ANNOTATION_FILE)

    if os.path.exists(ann_path_same):
        annotation_path = ann_path_same
    elif os.path.exists(ann_path_parent):
        annotation_path = ann_path_parent
    else:
        raise FileNotFoundError(f"Annotation file '{ANNOTATION_FILE}' not found in or above {IMAGES_DIR}")

    with open(annotation_path, 'r') as f:
        coco = json.load(f)

    images = coco['images']
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    n_test = n_total - n_train - n_val

    splits = {
        'train': images[:n_train],
        'val':   images[n_train:n_train + n_val],
        'test':  images[n_train + n_val:]
    }

    for split_name, split_images in splits.items():
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        make_dir(split_dir)

        # New annotation subset
        img_ids = {img['id'] for img in split_images}
        split_anns = [ann for ann in coco['annotations'] if ann['image_id'] in img_ids]

        split_json = {
            'info': coco.get('info', {}),
            'licenses': coco.get('licenses', []),
            'categories': coco['categories'],
            'images': split_images,
            'annotations': split_anns
        }

        with open(os.path.join(split_dir, ANNOTATION_FILE), 'w') as f:
            json.dump(split_json, f)

        # Copy corresponding images
        for img in split_images:
            file_name = os.path.basename(img['file_name'])
            src = os.path.join(IMAGES_DIR, file_name)
            dst = os.path.join(split_dir, file_name)

            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: file not found -> {src}")

        print(f"{split_name}: {len(split_images)} images saved.")

    print("Dataset split complete.")


if __name__ == "__main__":
    split_dataset()
