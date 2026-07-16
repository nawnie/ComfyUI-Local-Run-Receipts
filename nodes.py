"""Local-only ComfyUI output nodes for stable image run receipts."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import folder_paths
from PIL import Image
import numpy as np


KEY_SCHEMA = "local-run-receipts/key/v1"
RECEIPT_SCHEMA = "local-run-receipts/receipt/v1"
RUN_KEY_PATTERN = re.compile(r"^lrr1_[0-9a-f]{64}$")
PATH_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class RunReceiptError(ValueError):
    """A user-facing error that leaves existing receipt data untouched."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise RunReceiptError(f"parameters_json cannot contain {value}")


def _parse_json_object(value: str, field_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise RunReceiptError(f"{field_name} must be valid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise RunReceiptError(f"{field_name} must contain a JSON object")
    try:
        _canonical_json(parsed)
    except (TypeError, ValueError) as error:
        raise RunReceiptError(f"{field_name} contains a value that cannot be canonicalized") from error
    return parsed


def _validate_text(value: str, field_name: str, limit: int = 160) -> str:
    if not isinstance(value, str):
        raise RunReceiptError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise RunReceiptError(f"{field_name} cannot be empty")
    if len(normalized) > limit:
        raise RunReceiptError(f"{field_name} is limited to {limit} characters")
    return normalized


def _validate_parent_key(value: str) -> str:
    if not isinstance(value, str):
        raise RunReceiptError("parent_run_key must be text")
    normalized = value.strip()
    if not normalized:
        return ""
    if not RUN_KEY_PATTERN.fullmatch(normalized):
        raise RunReceiptError("parent_run_key must be empty or a Local Run Receipts key")
    return normalized


def _normalise_key_record(record: dict[str, Any]) -> dict[str, Any]:
    required = {"schema", "namespace", "label", "seed", "parameters"}
    allowed = required | {"parent_run_key"}
    keys = set(record)
    missing = sorted(required - keys)
    unknown = sorted(keys - allowed)
    if missing:
        raise RunReceiptError(f"canonical_record is missing: {', '.join(missing)}")
    if unknown:
        raise RunReceiptError(f"canonical_record has unsupported fields: {', '.join(unknown)}")
    if record["schema"] != KEY_SCHEMA:
        raise RunReceiptError(f"canonical_record must use {KEY_SCHEMA}")
    if not isinstance(record["seed"], int) or isinstance(record["seed"], bool):
        raise RunReceiptError("canonical_record seed must be an integer")
    if record["seed"] < 0:
        raise RunReceiptError("canonical_record seed cannot be negative")
    if not isinstance(record["parameters"], dict):
        raise RunReceiptError("canonical_record parameters must be a JSON object")

    normalized = {
        "schema": KEY_SCHEMA,
        "namespace": _validate_text(record["namespace"], "namespace", 80),
        "label": _validate_text(record["label"], "label"),
        "seed": record["seed"],
        "parameters": record["parameters"],
    }
    parent_run_key = _validate_parent_key(record.get("parent_run_key", ""))
    if parent_run_key:
        normalized["parent_run_key"] = parent_run_key
    try:
        _canonical_json(normalized)
    except (TypeError, ValueError) as error:
        raise RunReceiptError("canonical_record contains a value that cannot be canonicalized") from error
    return normalized


def _build_run_key(record: dict[str, Any]) -> tuple[str, str]:
    canonical_record = _canonical_json(record)
    run_key = f"lrr1_{_sha256(canonical_record.encode('utf-8'))}"
    return run_key, canonical_record


def _parse_and_check_record(run_key: str, canonical_record: str) -> dict[str, Any]:
    if not isinstance(run_key, str) or not RUN_KEY_PATTERN.fullmatch(run_key):
        raise RunReceiptError("run_key must be a Local Run Receipts key")
    if not isinstance(canonical_record, str):
        raise RunReceiptError("canonical_record must be text from Build Run Key")
    record = _normalise_key_record(_parse_json_object(canonical_record, "canonical_record"))
    expected_key, _ = _build_run_key(record)
    if run_key != expected_key:
        raise RunReceiptError("run_key does not match canonical_record")
    return record


def _safe_subfolder(value: str) -> str:
    if not isinstance(value, str):
        raise RunReceiptError("output_subfolder must be text")
    normalized = value.strip().replace("\\", "/")
    if not normalized or len(normalized) > 160:
        raise RunReceiptError("output_subfolder must be 1 to 160 characters")
    segments = normalized.split("/")
    if any(segment in {"", ".", ".."} or not PATH_SEGMENT_PATTERN.fullmatch(segment) for segment in segments):
        raise RunReceiptError("output_subfolder may contain only safe relative path segments")
    return "/".join(segments)


def _safe_prefix(value: str) -> str:
    if not isinstance(value, str):
        raise RunReceiptError("filename_prefix must be text")
    normalized = value.strip()
    if not PREFIX_PATTERN.fullmatch(normalized):
        raise RunReceiptError("filename_prefix may contain letters, numbers, dots, dashes, and underscores")
    return normalized


def _encode_image(image: Any) -> tuple[bytes, int, int]:
    pixels = image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[-1] not in {3, 4}:
        raise RunReceiptError("images must contain HxWx3 or HxWx4 image tensors")
    image_array = np.clip(255.0 * pixels[..., :3], 0, 255).astype(np.uint8)
    height, width = image_array.shape[:2]
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG", compress_level=4)
    return buffer.getvalue(), width, height


def _prompt_snapshot_hash(prompt: Any) -> str | None:
    if prompt is None:
        return None
    try:
        return _sha256(_canonical_json(prompt).encode("utf-8"))
    except (TypeError, ValueError) as error:
        raise RunReceiptError("prompt snapshot cannot be canonicalized") from error


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


class BuildLocalRunKey:
    """Builds an immutable identifier from explicit, user-declared run values."""

    CATEGORY = "output/local run receipts"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("run_key", "canonical_record")
    FUNCTION = "build"
    DESCRIPTION = "Builds a stable local run key from declared values. It does not inspect models or upload data."
    SEARCH_ALIASES = ["run receipt", "run key", "idempotent save", "image manifest"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "namespace": ("STRING", {"default": "local", "multiline": False}),
                "label": ("STRING", {"default": "SD 1.5 image", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9_007_199_254_740_991}),
                "parameters_json": ("STRING", {"default": "{\"sampler\":\"euler\",\"steps\":20}", "multiline": True}),
                "parent_run_key": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def build(
        self,
        namespace: str,
        label: str,
        seed: int,
        parameters_json: str,
        parent_run_key: str = "",
    ) -> tuple[str, str]:
        record: dict[str, Any] = {
            "schema": KEY_SCHEMA,
            "namespace": namespace,
            "label": label,
            "seed": seed,
            "parameters": _parse_json_object(parameters_json, "parameters_json"),
        }
        if parent_run_key.strip():
            record["parent_run_key"] = parent_run_key
        normalized = _normalise_key_record(record)
        return _build_run_key(normalized)


class CommitLocalRunImages:
    """Commits image bytes and a receipt under a run key without overwriting a prior run."""

    CATEGORY = "output/local run receipts"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "receipt_path")
    FUNCTION = "commit_images"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves local images and a receipt once. Repeating identical data returns ALREADY_IDENTICAL; changed data raises a conflict."
    SEARCH_ALIASES = ["run receipt", "commit image", "idempotent save", "image manifest"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to commit under the run key."}),
                "run_key": ("STRING", {"multiline": False}),
                "canonical_record": ("STRING", {"multiline": True}),
                "output_subfolder": ("STRING", {"default": "local-run-receipts", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "run", "multiline": False}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs: Any) -> float:
        """Force each output-node invocation to inspect the existing receipt."""
        return float("nan")

    def _receipt_response(
        self,
        status: str,
        run_key: str,
        receipt_path: str,
        image_entries: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "ui": {
                "images": image_entries,
                "local_run_receipts": [{
                    "status": status,
                    "run_key": run_key,
                    "receipt_path": receipt_path,
                }],
            },
            "result": (status, receipt_path),
        }

    def _check_existing(self, run_directory: Path, receipt_bytes: bytes, artifacts: list[dict[str, Any]]) -> str | None:
        if not run_directory.exists():
            return None
        if not run_directory.is_dir():
            raise RunReceiptError("run key path exists but is not a directory")
        receipt_file = run_directory / "receipt.json"
        if not receipt_file.exists():
            raise RunReceiptError("run key is incomplete and will not be overwritten")
        if receipt_file.read_bytes() != receipt_bytes:
            raise RunReceiptError("run key conflict: stored receipt differs from this run")
        for artifact in artifacts:
            artifact_path = run_directory / artifact["filename"]
            if not artifact_path.is_file() or _sha256(artifact_path.read_bytes()) != artifact["sha256"]:
                raise RunReceiptError("run key conflict: stored image bytes differ from this run")
        return "ALREADY_IDENTICAL"

    def commit_images(
        self,
        images: Any,
        run_key: str,
        canonical_record: str,
        output_subfolder: str = "local-run-receipts",
        filename_prefix: str = "run",
        prompt: Any = None,
    ) -> dict[str, Any]:
        record = _parse_and_check_record(run_key, canonical_record)
        safe_subfolder = _safe_subfolder(output_subfolder)
        safe_prefix = _safe_prefix(filename_prefix)

        encoded_images = [_encode_image(image) for image in images]
        if not encoded_images:
            raise RunReceiptError("images cannot be empty")

        artifacts: list[dict[str, Any]] = []
        image_bytes_by_filename: dict[str, bytes] = {}
        for index, (image_bytes, width, height) in enumerate(encoded_images, start=1):
            filename = f"{safe_prefix}-{index:03d}.png"
            image_bytes_by_filename[filename] = image_bytes
            artifacts.append({
                "filename": filename,
                "sha256": _sha256(image_bytes),
                "bytes": len(image_bytes),
                "width": width,
                "height": height,
            })

        receipt = {
            "schema": RECEIPT_SCHEMA,
            "run_key": run_key,
            "key_record": record,
            "prompt_sha256": _prompt_snapshot_hash(prompt),
            "artifacts": artifacts,
        }
        receipt_bytes = _canonical_json(receipt).encode("utf-8")

        output_root = Path(folder_paths.get_output_directory()).resolve()
        run_parent = (output_root / safe_subfolder).resolve()
        try:
            run_parent.relative_to(output_root)
        except ValueError as error:
            raise RunReceiptError("output_subfolder resolves outside the ComfyUI output directory") from error
        run_directory = run_parent / run_key
        receipt_path = f"{safe_subfolder}/{run_key}/receipt.json"
        image_entries = [
            {"filename": artifact["filename"], "subfolder": f"{safe_subfolder}/{run_key}", "type": "output"}
            for artifact in artifacts
        ]

        existing_status = self._check_existing(run_directory, receipt_bytes, artifacts)
        if existing_status:
            return self._receipt_response(existing_status, run_key, receipt_path, image_entries)

        run_parent.mkdir(parents=True, exist_ok=True)
        try:
            run_directory.mkdir()
        except FileExistsError:
            existing_status = self._check_existing(run_directory, receipt_bytes, artifacts)
            if existing_status:
                return self._receipt_response(existing_status, run_key, receipt_path, image_entries)
            raise RunReceiptError("run key could not be reserved")

        for artifact in artifacts:
            _atomic_write(run_directory / artifact["filename"], image_bytes_by_filename[artifact["filename"]])
        _atomic_write(run_directory / "receipt.json", receipt_bytes)
        return self._receipt_response("CREATED", run_key, receipt_path, image_entries)
