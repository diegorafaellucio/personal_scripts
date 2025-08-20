#!/usr/bin/env python3.11
"""
Download images from an S3 bucket prefix with concurrency and resume support.

Requirements:
  - Python 3.11 (preferred)
  - boto3 (pip install boto3)

Usage examples:
  # Using environment variables for AWS credentials
  export AWS_ACCESS_KEY_ID=...; export AWS_SECRET_ACCESS_KEY=...
  python3.11 download_s3_images.py \
      --bucket cogtive-computer-vision-prod \
      --prefix video_frames/ \
      --dest /home/diego/2TB/datasets/COGTIVE/KOVI/GENERAL/1.0/IMAGES \
      --max-files 25000 --workers 32 --resume

  # Or pass credentials via CLI flags (not recommended for shared terminals)
  python3.11 download_s3_images.py \
      --bucket cogtive-computer-vision-prod \
      --prefix video_frames/ \
      --dest /home/diego/2TB/datasets/COGTIVE/KOVI/GENERAL/1.0/IMAGES \
      --max-files 25000 --workers 32 --resume \
      --aws-access-key-id YOUR_KEY --aws-secret-access-key YOUR_SECRET

Notes:
  - The script preserves the folder structure under the provided prefix.
  - To speed up downloads, increase --workers (typical: 16-64).
  - By default, only common image extensions are downloaded; use --extensions "*" to download all file types.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import logging
import os
import sys
import threading
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError as e:
    print("This script requires boto3. Install it with: pip install boto3", file=sys.stderr)
    raise


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download up to N files from an S3 prefix.")

    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--prefix",
        default="",
        help="S3 prefix to list from (e.g., 'video_frames/'). Use empty for whole bucket.",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Local destination directory where files will be saved.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=25000,
        help="Maximum number of files to download.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of concurrent download workers.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already exist locally with the same size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help=(
            "Comma-separated list of file extensions to include (case-insensitive). "
            "Use '*' to include all files. Default: .jpg,.jpeg,.png,.bmp,.webp"
        ),
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region name (optional). If not provided, boto3 will auto-resolve.",
    )
    parser.add_argument(
        "--aws-access-key-id",
        default=None,
        help="AWS Access Key ID (optional; otherwise read from env/credentials).",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        default=None,
        help="AWS Secret Access Key (optional; otherwise read from env/credentials).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS profile name to use from local credentials file (optional).",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the selection order before downloading (off by default).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list the selected keys (no downloads).",
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Do not preserve the S3 prefix structure; flatten into dest directory.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N files (default: 1000).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args(argv)

    # Normalize and validate
    args.dest = os.path.abspath(os.path.expanduser(args.dest))
    if args.prefix and not args.prefix.endswith("/"):
        # Not strictly required, but common when preserving structure.
        args.prefix = args.prefix
    return args


def build_boto3_session(
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    region_name: Optional[str],
    profile: Optional[str],
):
    if profile:
        return boto3.Session(profile_name=profile, region_name=region_name)
    return boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )


def list_s3_keys(
    s3_client,
    bucket: str,
    prefix: str,
    max_files: int,
    extensions: Optional[List[str]] = None,
    randomize: bool = False,
) -> List[Tuple[str, int]]:
    """Return up to max_files (key, size) under the prefix, optionally filtered by extension."""
    paginator = s3_client.get_paginator("list_objects_v2")
    selected: List[Tuple[str, int]] = []

    def is_image(key: str) -> bool:
        if not extensions:
            return True
        if extensions == ["*"]:
            return True
        lk = key.lower()
        return any(lk.endswith(ext) for ext in extensions)

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                size = obj.get("Size", 0)
                # Skip folder placeholders and zero-sized keys
                if key.endswith("/") or size == 0:
                    continue
                if not is_image(key):
                    continue
                selected.append((key, size))
                if len(selected) >= max_files:
                    break
            if len(selected) >= max_files:
                break
    except (BotoCoreError, ClientError) as e:
        logging.error("Failed to list objects: %s", e)
        raise

    if randomize:
        import random

        random.shuffle(selected)
        selected = selected[:max_files]

    return selected


def local_path_for_key(dest_root: str, key: str, prefix: str, preserve_structure: bool) -> str:
    if preserve_structure:
        # Remove the leading prefix from the key when saving under dest
        rel = key[len(prefix) :] if prefix and key.startswith(prefix) else key
        rel = rel.lstrip("/")
        return os.path.join(dest_root, rel)
    else:
        # Flatten: use only the filename part
        fname = os.path.basename(key)
        return os.path.join(dest_root, fname)


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def download_one(
    s3_client,
    bucket: str,
    key: str,
    size: int,
    dest_path: str,
    resume: bool,
) -> Tuple[str, bool, Optional[str]]:
    """
    Download a single object. Returns (key, downloaded, error_message).
    downloaded is True if a download occurred, False if skipped by resume.
    """
    try:
        if resume and os.path.exists(dest_path):
            try:
                stat = os.stat(dest_path)
                if int(stat.st_size) == int(size):
                    return key, False, None  # skip
            except OSError:
                pass  # fallthrough to re-download
        ensure_parent_dir(dest_path)
        s3_client.download_file(bucket, key, dest_path)
        return key, True, None
    except (BotoCoreError, ClientError, OSError) as e:
        return key, False, str(e)


def download_many(
    s3_client,
    bucket: str,
    prefix: str,
    items: List[Tuple[str, int]],
    dest_root: str,
    resume: bool,
    workers: int,
    log_every: int,
    preserve_structure: bool,
) -> None:
    total = len(items)
    logging.info("Starting downloads: %d files -> %s", total, dest_root)

    # Thread-safe counters
    lock = threading.Lock()
    stats = {"done": 0, "skipped": 0, "errors": 0}

    def task(item: Tuple[str, int]):
        key, size = item
        dest_path = local_path_for_key(dest_root, key, prefix, preserve_structure)
        k, downloaded, err = download_one(s3_client, bucket, key, size, dest_path, resume)
        with lock:
            stats["done"] += 1
            if not downloaded and err is None:
                stats["skipped"] += 1
            if err is not None:
                stats["errors"] += 1
                logging.debug("Error on %s: %s", k, err)
            if stats["done"] % max(1, log_every) == 0:
                logging.info(
                    "Progress: %d/%d (skipped=%d, errors=%d)",
                    stats["done"],
                    total,
                    stats["skipped"],
                    stats["errors"],
                )

    os.makedirs(dest_root, exist_ok=True)

    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(task, items))

    logging.info(
        "Completed: %d files processed (skipped=%d, errors=%d)",
        stats["done"],
        stats["skipped"],
        stats["errors"],
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    # Prepare extensions list
    extensions: Optional[List[str]]
    if args.extensions.strip() == "*":
        extensions = ["*"]
    else:
        extensions = [e.strip().lower() for e in args.extensions.split(",") if e.strip()]

    try:
        session = build_boto3_session(
            aws_access_key_id=args.aws_access_key_id,
            aws_secret_access_key=args.aws_secret_access_key,
            region_name=args.region,
            profile=args.profile,
        )
        s3 = session.client("s3")
    except (BotoCoreError, ClientError) as e:
        logging.error("Failed to create S3 client: %s", e)
        return 2

    # List keys
    logging.info(
        "Listing up to %d files from s3://%s/%s",
        args.max_files,
        args.bucket,
        args.prefix,
    )
    try:
        items = list_s3_keys(
            s3_client=s3,
            bucket=args.bucket,
            prefix=args.prefix,
            max_files=args.max_files,
            extensions=extensions,
            randomize=args.randomize,
        )
    except Exception:
        return 3

    if not items:
        logging.warning("No matching files found.")
        return 0

    logging.info("Selected %d files.", len(items))

    if args.list_only or args.dry_run:
        for key, size in items[:50]:
            logging.info("Would download: %s (%d bytes)", key, size)
        if len(items) > 50:
            logging.info("... and %d more", len(items) - 50)
        return 0

    # Download
    try:
        download_many(
            s3_client=s3,
            bucket=args.bucket,
            prefix=args.prefix,
            items=items,
            dest_root=args.dest,
            resume=args.resume,
            workers=args.workers,
            log_every=args.log_every,
            preserve_structure=not args.no_structure,
        )
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
