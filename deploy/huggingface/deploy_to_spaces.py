"""Push this project to a Hugging Face Space (free CPU tier, no credit card).

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxx     # a Write token from
                                         # https://huggingface.co/settings/tokens
    python -m deploy.huggingface.deploy_to_spaces <your-username>/telco-churn-api

Creates the Space if it does not already exist, then uploads exactly what the
Docker build needs: this folder's Dockerfile and README.md, plus src/,
requirements.txt and the trained artifacts. Nothing else -- tests, the
notebook, and the AWS deploy scripts have no reason to be in the Space's git
history.

`make train` must have produced artifacts/model.joblib and
artifacts/model_metadata.json before this script is run; it refuses to push a
Space with no model rather than build one that returns 503 forever.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HF_DIR = Path(__file__).resolve().parent


def build_upload_dir(dest: Path) -> None:
    """Assemble the exact file set the Space's Dockerfile expects, under `dest`."""
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy(HF_DIR / "Dockerfile", dest / "Dockerfile")
    shutil.copy(HF_DIR / "README.md", dest / "README.md")
    shutil.copy(PROJECT_ROOT / "requirements.txt", dest / "requirements.txt")
    shutil.copytree(
        PROJECT_ROOT / "src",
        dest / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    model = PROJECT_ROOT / "artifacts" / "model.joblib"
    metadata = PROJECT_ROOT / "artifacts" / "model_metadata.json"
    if not model.exists() or not metadata.exists():
        print(
            f"error: {model} not found. Run `make train` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    artifacts_dir = dest / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    shutil.copy(model, artifacts_dir / "model.joblib")
    shutil.copy(metadata, artifacts_dir / "model_metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "space_id", help="e.g. yourname/telco-churn-api (owner/space-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="create the Space as private (still free on the CPU-basic tier)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Imported here, not at module load, so build_upload_dir can be tested
    # without huggingface_hub installed.
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "error: huggingface_hub is not installed. Run "
            "`pip install -r requirements-dev.txt` (or `pip install huggingface_hub`).",
            file=sys.stderr,
        )
        return 1

    api = HfApi()  # reads the HF_TOKEN environment variable
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        upload_dir = Path(tmp) / "space"
        build_upload_dir(upload_dir)
        api.upload_folder(
            repo_id=args.space_id,
            repo_type="space",
            folder_path=str(upload_dir),
            commit_message="Deploy telco-churn-api",
        )

    space_url = f"https://huggingface.co/spaces/{args.space_id}"
    print(f"\nPushed. Build progress: {space_url}")
    print(
        "Once the Space shows \"Running\", its app URL is on that page under "
        "\"Embed this Space\" -- usually https://<owner>-<space-name>.hf.space "
        "with underscores in the space name, if any, kept as-is."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
