"""Download the approved model snapshots for later offline application use."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download

from config.settings import MODELS_DIR


MODELS = {
    "embedding": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        MODELS_DIR / "embeddings",
    ),
    "nllb": (
        "facebook/nllb-200-distilled-600M",
        MODELS_DIR / "nllb",
    ),
}


def download_model(name: str, repository: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} from {repository} to {destination}")
    snapshot_download(
        repo_id=repository,
        local_dir=destination,
        local_dir_use_symlinks=False,
    )
    print(f"Finished {name}")


def main() -> None:
    for name, (repository, destination) in MODELS.items():
        download_model(name, repository, destination)


if __name__ == "__main__":
    main()
