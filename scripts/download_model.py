"""Download local embedding models into the model folder (service.json: model_dir).

Usage:
    python -m scripts.download_model                  # the default model
    python -m scripts.download_model --model bge-m3   # one configured alias
    python -m scripts.download_model --all            # every local model in config/models.json
    python -m scripts.download_model --force          # download again even if present

Models already on disk are skipped, so it is safe to run on every start.
API models (provider "openai") have nothing to download and are skipped.
"""
import argparse
import logging

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:  # API-only deployments do not install it
    HfApi = snapshot_download = None

from app.core.config import get_config, get_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_downloaded(target) -> bool:
    return target.is_dir() and any((target / name).exists() for name in ("config.json", "modules.json"))


def download(alias: str, config, token: str | None, force: bool) -> None:
    if not config.is_local(alias):
        logger.info("Model %s is served through an API, nothing to download", alias)
        return

    target = config.model_path(alias)
    name = config.models[alias].name

    if is_downloaded(target) and not force:
        logger.info("Model %s (%s) already at %s, skipping", alias, name, target)
        return

    if snapshot_download is None:
        raise SystemExit(
            f"Cannot download '{alias}': huggingface_hub is not installed. Install requirements-local.txt"
        )

    logger.info("Downloading %s (%s) to %s", alias, name, target)

    ignore = ["onnx/*", "openvino/*", "*.onnx", "*.msgpack", "*.h5"]
    files = HfApi().list_repo_files(name, token=token)
    # Skip the duplicate PyTorch .bin only when safetensors weights exist.
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("pytorch_model*.bin")

    snapshot_download(repo_id=name, local_dir=target, token=token, ignore_patterns=ignore)
    logger.info("Done: %s", alias)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", metavar="ALIAS", help="alias from config/models.json")
    selection.add_argument("--all", action="store_true", help="download every local model")
    parser.add_argument("--force", action="store_true", help="download even if already present")
    args = parser.parse_args(argv)

    config = get_config()
    secret = get_env().hf_token
    token = secret.get_secret_value() if secret else None

    if args.all:
        aliases = list(config.models)
    else:
        alias = args.model or config.service.default_model
        if alias not in config.models:
            raise SystemExit(f"Unknown model '{alias}'. Available: {', '.join(config.models)}")
        aliases = [alias]

    for alias in aliases:
        download(alias, config, token, args.force)


if __name__ == "__main__":
    main()
