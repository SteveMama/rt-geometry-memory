from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


DATASET_FILES = {
    "longmemeval_s_cleaned": {
        "repo_id": "xiaowu0162/longmemeval-cleaned",
        "repo_type": "dataset",
        "filename": "longmemeval_s_cleaned.json",
        "recommended_format": "longmemeval",
    },
    "longmemeval_m_cleaned": {
        "repo_id": "xiaowu0162/longmemeval-cleaned",
        "repo_type": "dataset",
        "filename": "longmemeval_m_cleaned.json",
        "recommended_format": "longmemeval",
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a public benchmark file for the Paper 3 Colab runs.")
    parser.add_argument(
        "--benchmark",
        choices=sorted(DATASET_FILES),
        default="longmemeval_s_cleaned",
        help="Benchmark file to download. The default is the recommended first public benchmark.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = DATASET_FILES[args.benchmark]
    downloaded = hf_hub_download(
        repo_id=config["repo_id"],
        repo_type=config["repo_type"],
        filename=config["filename"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, args.output)
    print(f"Downloaded {args.benchmark} to {args.output}")
    print(f"Recommended format: {config['recommended_format']}")


if __name__ == "__main__":
    main()
