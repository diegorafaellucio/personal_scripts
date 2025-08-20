import os
import shutil
import math
import random
from typing import List, Iterable


def fileGenerator(data: Iterable[str], file_name: str) -> None:
    with open(file_name, "w") as f:
        for path in data:
            f.write(f"{path}\n")


def _collect_images_from_dataset(dataset_path: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    images_dir = os.path.join(dataset_path, "IMAGES")
    if not os.path.isdir(images_dir):
        print(f"WARNING: Images path not found: {images_dir}")
        return []
    exts = tuple(e.lower() for e in exts)
    files = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f)) and os.path.splitext(f)[1].lower() in exts
    ]
    return files


def generateDatabase(dataset_paths: List[str],
                     data_output_path: str,
                     eval_ratio: float = 0.1,
                     test_ratio: float = 0.2,
                     train_ratio: float = 0.7,
                     seed: int = 42) -> str:
    """
    Create train.txt, test.txt, eval.txt from multiple dataset roots.

    Each dataset root must contain an 'IMAGES' folder with images.

    Args:
        dataset_paths: list of dataset root folders
        data_output_path: path to DATA folder or a parent folder (if it doesn't contain 'DATA', '/DATA' is appended)
        eval_ratio, test_ratio, train_ratio: split ratios (default 0.1/0.2/0.7)
        seed: RNG seed for reproducibility of shuffling

    Returns:
        The final DATA directory path where txt files were written.
    """
    # Resolve output DATA path
    if "DATA" in os.path.basename(os.path.normpath(data_output_path)):
        data_path = data_output_path
    else:
        data_path = os.path.join(data_output_path, "DATA")

    # Recreate DATA folder cleanly
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path, exist_ok=True)

    eval_file = os.path.join(data_path, "eval.txt")
    train_file = os.path.join(data_path, "train.txt")
    test_file = os.path.join(data_path, "test.txt")

    # Collect and deduplicate images
    all_images: List[str] = []
    for p in dataset_paths:
        print(f"Collecting from: {p}")
        imgs = _collect_images_from_dataset(p)
        print(f"  Found {len(imgs)} images")
        all_images.extend(imgs)

    # Deduplicate
    all_images = list(dict.fromkeys(all_images))

    total = len(all_images)
    print(f"Total unique images collected: {total}")

    if total == 0:
        # Write empty files and return
        fileGenerator([], eval_file)
        fileGenerator([], train_file)
        fileGenerator([], test_file)
        print(f"No images found. Created empty split files at: {data_path}")
        return data_path

    # Shuffle predictably
    rnd = random.Random(seed)
    rnd.shuffle(all_images)

    # Compute split counts (keep original behavior: floor eval/train, remainder to test)
    eval_amount = int(math.floor(total * eval_ratio))
    train_amount = int(math.floor(total * train_ratio))
    # Ensure non-negative remainder
    test_amount = max(0, total - eval_amount - train_amount)

    eval_data = all_images[:eval_amount]
    train_data = all_images[eval_amount:eval_amount + train_amount]
    test_data = all_images[eval_amount + train_amount:]

    # Persist
    fileGenerator(eval_data, eval_file)
    fileGenerator(train_data, train_file)
    fileGenerator(test_data, test_file)

    print(f"Generated files at: {data_path}")
    print(f"Eval:  {len(eval_data)}")
    print(f"Train: {len(train_data)}")
    print(f"Test:  {len(test_data)}")

    return data_path


if __name__ == "__main__":
    # Example based on your request
    generateDatabase([
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/9.0/',
        '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PGO/1.0/'
    ], '/home/diego/2TB/TREINOS/BM_9.0+PGO_1.0_NORMALIZADO/DATA/')
