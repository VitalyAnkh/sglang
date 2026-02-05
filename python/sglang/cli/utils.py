import contextlib
import json
import logging
import os
import subprocess
from functools import lru_cache

from sglang.srt.environ import envs
from sglang.utils import (
    has_diffusion_overlay_registry_match,
    is_known_non_diffusers_diffusion_model,
    load_diffusion_overlay_registry_from_env,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_overlay_registry() -> dict:
    return load_diffusion_overlay_registry_from_env()


def _is_overlay_diffusion_model(model_path: str) -> bool:
    return has_diffusion_overlay_registry_match(model_path, _load_overlay_registry())


def _is_registered_diffusion_model(model_path: str) -> bool:
    try:
        from sglang.multimodal_gen.registry import has_registered_diffusion_model_path
    except ImportError:
        # if diffusion dependencies are not installed
        return False

    return has_registered_diffusion_model_path(model_path)


def _is_diffusers_model_dir(model_dir: str) -> bool:
    """Check if a local directory contains a valid diffusers model_index.json."""
    config_path = os.path.join(model_dir, "model_index.json")
    if not os.path.exists(config_path):
        return False

    with open(config_path) as f:
        config = json.load(f)

    return "_diffusers_version" in config


def _get_gguf_architecture_local(path: str) -> str | None:
    if not os.path.isfile(path) or not path.lower().endswith(".gguf"):
        return None

    try:
        import gguf  # type: ignore
    except ImportError:
        return None

    try:
        reader = gguf.GGUFReader(path)
    except Exception:
        return None

    for key in ("general.architecture", "general.name"):
        field = reader.get_field(key)
        if field is None:
            continue
        with contextlib.suppress(Exception):
            value = field.contents()
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            text = str(value).strip()
            if text:
                return text.lower()

    return None


def _looks_like_qwen_image_gguf_local(path: str) -> bool:
    arch = _get_gguf_architecture_local(path)
    if arch == "qwen_image":
        return True

    low = os.path.basename(path).lower()
    return low.endswith(".gguf") and ("qwen" in low) and ("image" in low)


def _looks_like_qwen_image_gguf_repo(repo_id: str) -> bool:
    try:
        from huggingface_hub import model_info

        info = model_info(repo_id)
        gguf_meta = getattr(info, "gguf", None) or {}
        arch = str(gguf_meta.get("architecture", "")).strip().lower()
        if arch == "qwen_image":
            return True
    except Exception:
        pass

    low = repo_id.lower()
    return ("qwen" in low) and ("image" in low) and ("gguf" in low)


def get_is_diffusion_model(model_path: str) -> bool:
    """Detect whether model_path points to a diffusion model.

    For local directories, checks the filesystem directly.
    For HF/ModelScope model IDs, attempts to fetch only model_index.json and
    falls back to GGUF heuristics when appropriate.
    Returns False on any failure (network error, 404, offline mode, etc.)
    so that the caller falls through to the standard LLM server path.
    """
    if _is_overlay_diffusion_model(model_path):
        # short-circuit, if applicable for the overlay mechanism (diffusion-only)
        return True

    # Local file or directory: handle GGUF first, then fall back to diffusers/non-diffusers detection.
    if os.path.exists(model_path):
        if os.path.isfile(model_path):
            if _looks_like_qwen_image_gguf_local(model_path):
                logger.info("Qwen-Image GGUF model detected (local file).")
                return True
            return False

        if _is_diffusers_model_dir(model_path):
            return True

        gguf_candidates = [
            os.path.join(model_path, name)
            for name in os.listdir(model_path)
            if name.lower().endswith(".gguf")
        ]
        for cand in gguf_candidates[:3]:
            if _looks_like_qwen_image_gguf_local(cand):
                logger.info("Qwen-Image GGUF model detected (local dir).")
                return True

        return is_known_non_diffusers_diffusion_model(model_path)

    if is_known_non_diffusers_diffusion_model(model_path):
        return True

    if _is_registered_diffusion_model(model_path):
        return True

    # Remote model id: try diffusers model_index.json first, then GGUF heuristics.
    try:
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope import model_file_download

            file_path = model_file_download(
                model_id=model_path, file_path="model_index.json"
            )
        else:
            from huggingface_hub import hf_hub_download

            file_path = hf_hub_download(repo_id=model_path, filename="model_index.json")

        return _is_diffusers_model_dir(os.path.dirname(file_path))
    except Exception as e:
        logger.debug("Failed to auto-detect diffusion model for %s: %s", model_path, e)

    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(model_path)
        if any(f.lower().endswith(".gguf") for f in files) and _looks_like_qwen_image_gguf_repo(
            model_path
        ):
            logger.info("Qwen-Image GGUF model detected (HF repo).")
            return True
    except Exception:
        pass

    return False


def get_model_path(extra_argv):
    # Find the model_path argument
    model_path = None
    for i, arg in enumerate(extra_argv):
        if arg == "--model-path":
            if i + 1 < len(extra_argv):
                model_path = extra_argv[i + 1]
                break
        elif arg.startswith("--model-path="):
            model_path = arg.split("=", 1)[1]
            break

    if model_path is None:
        # Fallback for --help or other cases where model-path is not provided
        if any(h in extra_argv for h in ["-h", "--help"]):
            raise Exception(
                "Usage: sglang serve --model-path <model-name-or-path> [additional-arguments]\n\n"
                "This command can launch either a standard language model server or a diffusion model server.\n"
                "The server type is determined by the --model-path.\n"
            )
        else:
            raise Exception(
                "Error: --model-path is required. "
                "Please provide the path to the model."
            )
    return model_path


@lru_cache(maxsize=1)
def get_git_commit_hash() -> str:
    try:
        commit_hash = os.environ.get("SGLANG_GIT_COMMIT")
        if not commit_hash:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .strip()
                .decode("utf-8")
            )
        _CACHED_COMMIT_HASH = commit_hash
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        _CACHED_COMMIT_HASH = "N/A"
        return "N/A"
