"""Utility helpers for validating image inputs."""

import base64
import binascii
import os
from collections.abc import Iterable

from utils.file_types import IMAGES, get_image_mime_type

DEFAULT_MAX_IMAGE_SIZE_MB = 20.0

__all__ = ["DEFAULT_MAX_IMAGE_SIZE_MB", "validate_image"]


def _valid_mime_types() -> Iterable[str]:
    """Return the MIME types permitted by the IMAGES whitelist."""
    return (get_image_mime_type(ext) for ext in IMAGES)


def validate_image(image_path: str, max_size_mb: float = None) -> tuple[bytes, str]:
    """Validate a user-supplied image path or data URL.

    Args:
        image_path: Either a filesystem path or a data URL.
        max_size_mb: Optional size limit (defaults to ``DEFAULT_MAX_IMAGE_SIZE_MB``).

    Returns:
        A tuple ``(image_bytes, mime_type)`` ready for upstream providers.

    Raises:
        ValueError: When the image is missing, malformed, or exceeds limits.
    """
    if max_size_mb is None:
        max_size_mb = DEFAULT_MAX_IMAGE_SIZE_MB

    if image_path.startswith("data:"):
        return _validate_data_url(image_path, max_size_mb)

    return _validate_file_path(image_path, max_size_mb)


def _validate_data_url(image_data_url: str, max_size_mb: float) -> tuple[bytes, str]:
    """Validate a data URL and return image bytes plus MIME type."""
    try:
        header, data = image_data_url.split(",", 1)
        mime_type = header.split(";")[0].split(":")[1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid data URL format: {exc}")

    valid_mime_types = list(_valid_mime_types())
    if mime_type not in valid_mime_types:
        raise ValueError(
            "Unsupported image type: {mime}. Supported types: {supported}".format(
                mime=mime_type, supported=", ".join(valid_mime_types)
            )
        )

    try:
        image_bytes = base64.b64decode(data)
    except binascii.Error as exc:
        raise ValueError(f"Invalid base64 data: {exc}")

    _validate_size(image_bytes, max_size_mb)
    return image_bytes, mime_type


def _validate_file_path(file_path: str, max_size_mb: float) -> tuple[bytes, str]:
    """Validate an image loaded from the filesystem."""
    try:
        with open(file_path, "rb") as handle:
            image_bytes = handle.read()
    except FileNotFoundError:
        raise ValueError(f"Image file not found: {file_path}")
    except OSError as exc:
        raise ValueError(f"Failed to read image file: {exc}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in IMAGES:
        raise ValueError(
            "Unsupported image format: {ext}. Supported formats: {supported}".format(
                ext=ext, supported=", ".join(sorted(IMAGES))
            )
        )

    mime_type = get_image_mime_type(ext)
    _validate_size(image_bytes, max_size_mb)
    return image_bytes, mime_type


def _validate_size(image_bytes: bytes, max_size_mb: float) -> None:
    """Ensure the image does not exceed the configured size limit."""
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Image too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")
