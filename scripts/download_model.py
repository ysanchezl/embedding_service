"""Download the embedding model once into MODEL_DIR.

Usage: python -m scripts.download_model [--force]

Does nothing if the model is already there, so it is safe to run on every start.
"""
import argparse
import logging

from huggingface_hub import HfApi, snapshot_download

from app.services.embedding_service import model_path, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_downloaded(target) -> bool:
    return target.is_dir() and any((target / name).exists() for name in ("config.json", "modules.json"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="download even if already present")
    args = parser.parse_args(argv)

    if not settings.model_name:
        raise SystemExit("MODEL_NAME is not set")

    target = model_path()
    if is_downloaded(target) and not args.force:
        logger.info("Model %s already at %s, skipping download", settings.model_name, target)
        return

    logger.info("Downloading %s to %s", settings.model_name, target)

    ignore = ["onnx/*", "openvino/*", "*.onnx", "*.msgpack", "*.h5"]
    files = HfApi().list_repo_files(settings.model_name, token=settings.hf_token)
    # Skip the duplicate PyTorch .bin only when safetensors weights exist.
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("pytorch_model*.bin")

    snapshot_download(
        repo_id=settings.model_name,
        local_dir=target,
        token=settings.hf_token,
        ignore_patterns=ignore,
    )
    logger.info("Done")


if __name__ == "__main__":
    main()
