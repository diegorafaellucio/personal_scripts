#!/usr/bin/env python3
import csv
import os
import sys
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

 


def sniff_delimiter(csv_path: str) -> str:
    with open(csv_path, "rb") as f:
        sample = f.read(2048)
    try:
        sample_text = sample.decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # Fallback to comma
        return ","


def ensure_gsutil_available() -> None:
    if shutil.which("gsutil") is None:
        print("ERROR: 'gsutil' not found in PATH. Please install and authenticate (gcloud init / auth login).", file=sys.stderr)
        sys.exit(1)


def sanitize_subdir(name: str) -> str:
    # Keep as-is but avoid path traversal characters
    name = name.strip().replace("\\", "_")
    # Prevent absolute/relative injection
    name = name.replace("..", "_")
    return name


def prepare_tasks(
    csv_path: str,
    delimiter: str,
    image_col: str,
    acabamento_col: str,
    src_prefix: str,
    gcs_prefix: str,
    out_root: str,
) -> List[Tuple[str, str]]:
    """
    Returns list of (gcs_uri, local_dest_file)
    """
    tasks: List[Tuple[str, str]] = []
    missing_cols = 0
    invalid_rows = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if image_col not in reader.fieldnames or acabamento_col not in reader.fieldnames:
            print(f"ERROR: CSV missing required columns. Found: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)

        for row in reader:
            img_path = (row.get(image_col) or "").strip()
            acabamento = (row.get(acabamento_col) or "").strip()

            if not img_path or not acabamento:
                invalid_rows += 1
                continue

            if src_prefix not in img_path:
                # Not matching expected pattern; skip but warn
                invalid_rows += 1
                print(f"WARNING: Row image path doesn't contain src-prefix. Skipping. image='{img_path}'", file=sys.stderr)
                continue

            # Build GCS URI robustly to ensure exactly one slash between prefix and remainder
            remainder = img_path[len(src_prefix):]
            gcs_uri = gcs_prefix.rstrip('/') + '/' + remainder.lstrip('/')

            filename = os.path.basename(gcs_uri)
            if not filename:
                invalid_rows += 1
                print(f"WARNING: Could not extract filename from '{gcs_uri}'. Skipping.", file=sys.stderr)
                continue

            acabamento_dir = sanitize_subdir(acabamento)
            dest_dir = os.path.join(out_root, acabamento_dir, "IMAGES")
            dest_file = os.path.join(dest_dir, filename)
            tasks.append((gcs_uri, dest_file))

    if invalid_rows:
        print(f"INFO: Skipped {invalid_rows} row(s) due to missing/invalid values or unmatched prefix.")

    return tasks


def download_one(gcs_uri: str, dest_file: str, dry_run: bool, force: bool, retries: int = 2, show_progress: bool = False) -> Tuple[str, str, bool, Optional[str]]:
    """
    Returns (gcs_uri, dest_file, success, error_message)
    """
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)

    if not force and os.path.exists(dest_file):
        return (gcs_uri, dest_file, True, None)  # treat as success (skipped)

    if dry_run:
        print(f"[DRY-RUN] gsutil cp '{gcs_uri}' '{dest_file}'")
        return (gcs_uri, dest_file, True, None)

    # Use gsutil cp with retries
    attempts = 0
    last_err: Optional[str] = None
    while attempts <= max(0, retries):
        try:
            # Ensure directory each attempt (in case of concurrent races)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)

            # Prefer copying into the destination directory to reduce transient file creation issues
            dest_dir = os.path.dirname(dest_file)
            cmd = ["gsutil", "cp", gcs_uri, dest_dir + "/"]
            # If show_progress, let gsutil write directly to terminal to show progress bars
            stdout = None if show_progress else subprocess.PIPE
            stderr = None if show_progress else subprocess.PIPE
            proc = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True)
            if proc.returncode == 0:
                # Verify existence of the file after copy
                if os.path.exists(dest_file):
                    return (gcs_uri, dest_file, True, None)
                # If not immediately visible, brief wait and re-check once
                time.sleep(0.1)
                if os.path.exists(dest_file):
                    return (gcs_uri, dest_file, True, None)
                last_err = "File not found after successful gsutil cp"
                attempts += 1
                continue

            err_msg = (proc.stderr or "").strip() if not show_progress else "gsutil returned non-zero exit code"
            last_err = err_msg
            # Retry on directory-related errors
            if "No such file or directory" in err_msg or "OSError" in err_msg:
                attempts += 1
                # Exponential backoff
                time.sleep(min(5.0, 0.5 * (2 ** (attempts - 1))))
                continue
            # Non-retryable error
            break
        except Exception as e:
            last_err = str(e)
            if "No such file or directory" in last_err:
                attempts += 1
                time.sleep(min(5.0, 0.5 * (2 ** (attempts - 1))))
                continue
            break

    return (gcs_uri, dest_file, False, last_err or "Unknown gsutil error")


def group_tasks_by_dest_dir(tasks: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for gcs_uri, dest_file in tasks:
        dest_dir = os.path.dirname(dest_file)
        groups[dest_dir].append((gcs_uri, dest_file))
    return groups


def chunked(seq: List[Tuple[str, str]], size: int) -> List[List[Tuple[str, str]]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def bulk_copy_batch(dest_dir: str,
                    items: List[Tuple[str, str]],
                    dry_run: bool,
                    force: bool,
                    retries: int = 2,
                    show_progress: bool = False) -> Tuple[int, int, int, List[Tuple[str, str]], Optional[str]]:
    """
    Copy many GCS URIs into a destination directory using a single gsutil invocation.
    Returns (attempted, succeeded, failed, failed_items, last_error)
    """
    if not items:
        return (0, 0, 0, [], None)

    os.makedirs(dest_dir, exist_ok=True)

    attempted = len(items)

    # Dry-run prints a summary and command preview
    if dry_run:
        print(f"[DRY-RUN] gsutil -m cp {'-n ' if not force else ''}-I '{dest_dir}/'  # {len(items)} files")
        return (attempted, len(items), 0, [], None)

    remaining = items[:]
    last_err: Optional[str] = None
    attempt_idx = 0
    while remaining and attempt_idx <= max(0, retries):
        uris = "\n".join([u for u, _ in remaining]) + "\n"
        cmd = ["gsutil", "-m", "cp"]
        if not force:
            cmd.append("-n")
        cmd.extend(["-I", dest_dir + "/"])
        try:
            # If show_progress, stream stdout/stderr to terminal so gsutil can render progress bars
            stdout = None if show_progress else subprocess.PIPE
            stderr = None if show_progress else subprocess.PIPE
            proc = subprocess.run(cmd, input=uris, stdout=stdout, stderr=stderr, text=True)
            if proc.returncode != 0:
                if show_progress:
                    last_err = "gsutil returned non-zero exit code"
                else:
                    last_err = (proc.stderr or proc.stdout or "").strip() or "Unknown gsutil error"
        except Exception as e:
            last_err = str(e)

        # Check which files actually made it
        not_found: List[Tuple[str, str]] = []
        for gcs_uri, dest_file in remaining:
            if not os.path.exists(dest_file):
                not_found.append((gcs_uri, dest_file))

        if not not_found:
            return (attempted, attempted, 0, [], None)

        # Backoff and retry only missing ones
        attempt_idx += 1
        if attempt_idx <= retries:
            time.sleep(min(5.0, 0.5 * (2 ** (attempt_idx - 1))))
            remaining = not_found
        else:
            remaining = not_found
            break

    failed = len(remaining)
    succeeded = attempted - failed
    return (attempted, succeeded, failed, remaining, last_err)


def main(
    csv_path: str,
    src_prefix: str = "/data/local-files?d=tracebeef_images",
    gcs_prefix: str = "gs://repo-prod-ecoiabeef-imagens",
    out_root: str = "/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/MSO/6.0/",
    image_col: str = "image",
    acabamento_col: str = "ACABAMENTO",
    concurrency: int = 12,
    retries: int = 4,
    dry_run: bool = False,
    force: bool = False,
    bulk_mode: bool = True,
    batch_size: int = 250,
    show_progress: bool = False,
):
    """Programmatic entrypoint to run the downloader without relying on CLI parsing."""
    ensure_gsutil_available()

    delimiter = sniff_delimiter(csv_path)
    print(f"INFO: Detected CSV delimiter: '{delimiter}'")

    tasks = prepare_tasks(
        csv_path=csv_path,
        delimiter=delimiter,
        image_col=image_col,
        acabamento_col=acabamento_col,
        src_prefix=src_prefix,
        gcs_prefix=gcs_prefix,
        out_root=out_root,
    )

    if not tasks:
        print("No tasks to execute. Exiting.")
        return

    # Optionally pre-skip existing files when not forcing overwrite
    if not force and not dry_run:
        before = len(tasks)
        tasks = [(u, d) for (u, d) in tasks if not os.path.exists(d)]
        skipped = before - len(tasks)
        if skipped:
            print(f"INFO: Skipping {skipped} file(s) already present. Remaining: {len(tasks)}")

    print(f"INFO: Prepared {len(tasks)} file(s) to process.")
    attempted = 0
    succeeded = 0
    failed = 0

    if bulk_mode:
        groups = group_tasks_by_dest_dir(tasks)
        batches: List[Tuple[str, List[Tuple[str, str]]]] = []
        for dest_dir, items in groups.items():
            for chunk in chunked(items, max(1, batch_size)):
                batches.append((dest_dir, chunk))

        print(f"INFO: Created {len(batches)} batch(es) across {len(groups)} destination folder(s). Concurrency={concurrency} BatchSize={batch_size}")

        def run_batch(dest_dir: str, chunk: List[Tuple[str, str]]):
            return bulk_copy_batch(dest_dir, chunk, dry_run, force, retries, show_progress)

        if show_progress:
            # Run sequentially to avoid interleaving multiple progress bars
            total = sum(len(i) for _, i in batches)
            for idx, (dest_dir, chunk) in enumerate(batches, start=1):
                print(f"INFO: Starting batch {idx}/{len(batches)} -> {dest_dir} ({len(chunk)} files)")
                att, okc, flc, failed_items, err = run_batch(dest_dir, chunk)
                attempted += att
                succeeded += okc
                failed += flc
                if flc and err:
                    print(f"WARNING: Batch had failures: {err}", file=sys.stderr)
                print(f"Progress: {attempted}/{total} | ok={succeeded} | failed={failed}")
        else:
            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
                futures = [ex.submit(run_batch, dest_dir, chunk) for dest_dir, chunk in batches]
                for fut in as_completed(futures):
                    att, okc, flc, failed_items, err = fut.result()
                    attempted += att
                    succeeded += okc
                    failed += flc
                    if flc and err:
                        # Log only batch-level error; missing files will be summarized at the end
                        print(f"WARNING: Batch had failures: {err}", file=sys.stderr)

                    if (attempted % 500 == 0) or (attempted == sum(len(i) for _, i in batches)):
                        print(f"Progress: {attempted}/{sum(len(i) for _, i in batches)} | ok={succeeded} | failed={failed}")
    else:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures = [ex.submit(download_one, gcs_uri, dest, dry_run, force, retries, show_progress) for gcs_uri, dest in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                gcs_uri, dest_file, ok, err = fut.result()
                attempted += 1
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                    print(f"ERROR: Failed to download '{gcs_uri}' -> '{dest_file}': {err}", file=sys.stderr)

                if attempted % 50 == 0 or attempted == len(tasks):
                    print(f"Progress: {attempted}/{len(tasks)} | ok={succeeded} | failed={failed}")

    print("DONE")
    print(f"Attempted: {attempted}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    # Editable configuration. Set CSV_PATH as env var or edit below values.
    CSV_PATH = '/home/diego/2TB/novo_treino_modelo_unico/PRN_ACAB_AGOS/project-522-at-2025-08-20-07-41-54176c65.csv'

    # Defaults aligned with ECOTRACE setup
    SRC_PREFIX = "/data/local-files?d=suporte_cliente_datasets"
    GCS_PREFIX = "gs://repo-prod-label-studio"
    OUT_ROOT = "/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PRN/2.0"
    IMAGE_COL = "image"
    ACABAMENTO_COL = "ACABAMENTO"
    CONCURRENCY = 32
    RETRIES = 4
    DRY_RUN = False
    FORCE = False
    BULK_MODE = True
    BATCH_SIZE = 250
    SHOW_PROGRESS = True

    main(
        csv_path=CSV_PATH,
        src_prefix=SRC_PREFIX,
        gcs_prefix=GCS_PREFIX,
        out_root=OUT_ROOT,
        image_col=IMAGE_COL,
        acabamento_col=ACABAMENTO_COL,
        concurrency=CONCURRENCY,
        retries=RETRIES,
        dry_run=DRY_RUN,
        force=FORCE,
        bulk_mode=BULK_MODE,
        batch_size=BATCH_SIZE,
        show_progress=SHOW_PROGRESS,
    )