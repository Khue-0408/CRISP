"""Deterministic paired source-view geometry for the BWCR control.

This module only constructs paired source-training views and records enough
geometry to align future tensors back to the canonical 352 x 352 source frame.
It deliberately contains no boundary-distance field, consistency loss, model
forward, trainer dispatch, or target-time behavior.

Forward geometry follows the existing CRISP training order::

    horizontal flip -> vertical flip -> rotation/scale affine

Inverse alignment applies the mathematical reverse::

    inverse affine -> vertical flip -> horizontal flip

The matrix implementation composes those operations explicitly, so applying a
stored record never depends on random state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
from typing import Any, Literal, Mapping

import torch
import torch.nn.functional as F


CANONICAL_SOURCE_SIZE = (352, 352)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BWCRAugmentationContract:
    """Author-approved current-CRISP augmentation distribution."""

    image_size: tuple[int, int] = CANONICAL_SOURCE_SIZE
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotation_degrees: tuple[float, float] = (-15.0, 15.0)
    scale_range: tuple[float, float] = (0.75, 1.25)
    translation_pixels: tuple[float, float] = (0.0, 0.0)
    shear_degrees: tuple[float, float] = (0.0, 0.0)
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    saturation_range: tuple[float, float] = (0.9, 1.1)
    hue_range: tuple[float, float] = (-0.02, 0.02)
    gaussian_blur_probability: float = 0.10
    gaussian_blur_radius: tuple[float, float] = (0.1, 1.0)
    normalize_mean: tuple[float, ...] = _IMAGENET_MEAN
    normalize_std: tuple[float, ...] = _IMAGENET_STD

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class GeometryTransformRecord:
    """All sampled values required to replay and invert one view's geometry."""

    horizontal_flip: bool
    vertical_flip: bool
    rotation_degrees: float
    scale: float
    translation_pixels: tuple[float, float] = (0.0, 0.0)
    shear_degrees: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class AppearanceTransformRecord:
    """All sampled values required to replay one view's appearance transforms."""

    brightness_factor: float
    contrast_factor: float
    saturation_factor: float
    hue_factor: float
    gaussian_blur: bool
    gaussian_blur_radius: float | None


@dataclass(frozen=True)
class BWCRViewTransformRecord:
    """Serializable deterministic identity and sampled transform parameters."""

    rng_identity: str
    rng_seed: int
    geometry: GeometryTransformRecord
    appearance: AppearanceTransformRecord

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class BWCRSourceView:
    """One augmented source view and its replayable spatial metadata."""

    image: torch.Tensor
    mask: torch.Tensor
    forward_validity: torch.Tensor
    record: BWCRViewTransformRecord


@dataclass(frozen=True)
class BWCRSourceViewPair:
    """Two independently sampled views of one canonical source sample."""

    view_0: BWCRSourceView
    view_1: BWCRSourceView


def _pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return scalar, scalar
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"{name} must be a scalar or length-2 sequence, got {value!r}.")


def _require_close(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, tuple):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(
                f"{name}={actual!r} violates the current CRISP contract {expected!r}."
            )
        actual_values = tuple(float(value) for value in actual)
        if any(
            not math.isclose(a, e, rel_tol=0.0, abs_tol=1e-12)
            for a, e in zip(actual_values, expected)
        ):
            raise ValueError(
                f"{name}={actual_values!r} violates the current CRISP contract {expected!r}."
            )
        return
    if isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(f"{name}={actual!r} violates the current CRISP contract {expected!r}.")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name}={actual!r} violates the current CRISP contract {expected!r}.")


def current_crisp_bwcr_contract(config: Mapping[str, Any]) -> BWCRAugmentationContract:
    """Validate and materialize the locked paired-view augmentation contract.

    The paired control samples from the current CRISP augmentation family and
    ranges. This validator prevents a future control config from silently
    substituting the native BWCR grayscale-MRI augmentation distribution.
    """
    source = config.get("source_data", config)
    size = source.get("image_size", source.get("resize"))
    size_pair = (int(size), int(size)) if isinstance(size, int) else tuple(size or ())
    if size_pair != CANONICAL_SOURCE_SIZE:
        raise ValueError(
            f"BWCR paired views require canonical image_size={CANONICAL_SOURCE_SIZE}, "
            f"got {size_pair!r}."
        )

    _require_close("random_hflip", source.get("random_hflip"), True)
    _require_close("random_vflip", source.get("random_vflip"), True)
    _require_close("random_rotate_degrees", source.get("random_rotate_degrees"), 15.0)
    _require_close("random_scale_range", source.get("random_scale_range"), (0.75, 1.25))

    jitter = source.get("color_jitter") or {}
    for key, expected in (
        ("brightness", 0.10),
        ("contrast", 0.10),
        ("saturation", 0.10),
        ("hue", 0.02),
    ):
        _require_close(f"color_jitter.{key}", jitter.get(key), expected)

    blur = source.get("random_gaussian_blur") or {}
    probability = blur.get("probability", blur.get("p"))
    _require_close("random_gaussian_blur.probability", probability, 0.10)
    _require_close("random_gaussian_blur.sigma", blur.get("sigma"), (0.1, 1.0))
    _require_close("normalize_mean", source.get("normalize_mean"), _IMAGENET_MEAN)
    _require_close("normalize_std", source.get("normalize_std"), _IMAGENET_STD)
    return BWCRAugmentationContract()


def bwcr_view_rng_identity(
    experiment_seed: int,
    epoch: int,
    dataset_name: str,
    image_id: str,
    view_index: int,
) -> tuple[str, int]:
    """Return a stable digest identity and process-independent RNG seed."""
    if epoch < 0:
        raise ValueError("BWCR view epoch must be non-negative.")
    if view_index < 0:
        raise ValueError("BWCR view index must be non-negative.")
    if not dataset_name or not image_id:
        raise ValueError("BWCR paired views require dataset_name and image_id.")
    payload = (
        f"bwcr-source-view-v1\0{int(experiment_seed)}\0{int(epoch)}\0"
        f"{dataset_name}\0{image_id}\0{int(view_index)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return digest.hex(), int.from_bytes(digest[:8], "big") % (2**63)


def sample_bwcr_view_record(
    contract: BWCRAugmentationContract,
    *,
    experiment_seed: int,
    epoch: int,
    dataset_name: str,
    image_id: str,
    view_index: int,
) -> BWCRViewTransformRecord:
    """Sample one view without consuming global or another view's RNG state."""
    identity, seed = bwcr_view_rng_identity(
        experiment_seed, epoch, dataset_name, image_id, view_index
    )
    rng = random.Random(seed)
    geometry = GeometryTransformRecord(
        horizontal_flip=rng.random() < contract.horizontal_flip_probability,
        vertical_flip=rng.random() < contract.vertical_flip_probability,
        rotation_degrees=rng.uniform(*contract.rotation_degrees),
        scale=rng.uniform(*contract.scale_range),
        translation_pixels=contract.translation_pixels,
        shear_degrees=contract.shear_degrees,
    )
    blur = rng.random() < contract.gaussian_blur_probability
    appearance = AppearanceTransformRecord(
        brightness_factor=rng.uniform(*contract.brightness_range),
        contrast_factor=rng.uniform(*contract.contrast_range),
        saturation_factor=rng.uniform(*contract.saturation_range),
        hue_factor=rng.uniform(*contract.hue_range),
        gaussian_blur=blur,
        gaussian_blur_radius=(
            rng.uniform(*contract.gaussian_blur_radius) if blur else None
        ),
    )
    return BWCRViewTransformRecord(identity, seed, geometry, appearance)


def _homogeneous_matrix(
    geometry: GeometryTransformRecord,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map canonical normalized coordinates to augmented-view coordinates."""
    matrix = torch.eye(3, device=device, dtype=dtype)

    if geometry.horizontal_flip:
        horizontal = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        matrix = horizontal @ matrix
    if geometry.vertical_flip:
        vertical = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        matrix = vertical @ matrix

    angle = math.radians(geometry.rotation_degrees)
    shear_x = math.tan(math.radians(geometry.shear_degrees[0]))
    shear_y = math.tan(math.radians(geometry.shear_degrees[1]))
    cosine, sine = math.cos(angle), math.sin(angle)
    scale = geometry.scale
    if scale <= 0.0:
        raise ValueError("Geometry scale must be positive.")

    scale_matrix = torch.tensor(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    shear_matrix = torch.tensor(
        [[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    # Positive angles follow image-coordinate counter-clockwise convention.
    rotation = torch.tensor(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    tx = 2.0 * float(geometry.translation_pixels[0]) / float(width)
    ty = 2.0 * float(geometry.translation_pixels[1]) / float(height)
    translation = torch.tensor(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    affine = translation @ rotation @ shear_matrix @ scale_matrix
    return affine @ matrix


def _warp(
    tensor: torch.Tensor,
    geometry: GeometryTransformRecord,
    *,
    inverse_alignment: bool,
    interpolation: Literal["bilinear", "nearest"],
) -> torch.Tensor:
    if tensor.ndim not in (3, 4):
        raise ValueError("Spatial tensors must have shape [C,H,W] or [B,C,H,W].")
    squeeze = tensor.ndim == 3
    batch = tensor.unsqueeze(0) if squeeze else tensor
    if not torch.is_floating_point(batch):
        batch = batch.float()
    _, _, height, width = batch.shape
    forward = _homogeneous_matrix(
        geometry, height, width, device=batch.device, dtype=batch.dtype
    )
    # grid_sample maps output positions to input positions. Forward view
    # generation therefore samples with F^-1; inverse alignment samples the
    # augmented view at F(canonical_position).
    sampling = forward if inverse_alignment else torch.linalg.inv(forward)
    theta = sampling[:2].unsqueeze(0).expand(batch.shape[0], -1, -1)
    grid = F.affine_grid(theta, batch.shape, align_corners=False)
    warped = F.grid_sample(
        batch,
        grid,
        mode=interpolation,
        padding_mode="zeros",
        align_corners=False,
    )
    return warped.squeeze(0) if squeeze else warped


def apply_forward_geometry(
    tensor: torch.Tensor,
    geometry: GeometryTransformRecord,
    *,
    interpolation: Literal["bilinear", "nearest"],
) -> torch.Tensor:
    """Apply the stored canonical-to-view geometry."""
    return _warp(
        tensor, geometry, inverse_alignment=False, interpolation=interpolation
    )


def inverse_align_bwcr_tensor(
    tensor: torch.Tensor,
    geometry: GeometryTransformRecord,
    *,
    interpolation: Literal["bilinear", "nearest"] = "bilinear",
) -> torch.Tensor:
    """Align a view-frame tensor back to canonical coordinates."""
    return _warp(
        tensor, geometry, inverse_alignment=True, interpolation=interpolation
    )


def _rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.shape[0] == 1:
        return image
    if image.shape[0] != 3:
        raise ValueError("BWCR appearance transforms require one or three channels.")
    weights = image.new_tensor((0.2989, 0.5870, 0.1140)).view(3, 1, 1)
    return (image * weights).sum(dim=0, keepdim=True)


def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    red, green, blue = image.unbind(dim=0)
    value, max_index = image.max(dim=0)
    minimum = image.min(dim=0).values
    delta = value - minimum
    saturation = torch.where(value > 0, delta / value.clamp_min(1e-12), torch.zeros_like(value))
    safe_delta = delta.clamp_min(1e-12)
    hue = torch.zeros_like(value)
    hue = torch.where(max_index == 0, ((green - blue) / safe_delta) % 6.0, hue)
    hue = torch.where(max_index == 1, (blue - red) / safe_delta + 2.0, hue)
    hue = torch.where(max_index == 2, (red - green) / safe_delta + 4.0, hue)
    hue = torch.where(delta > 0, hue / 6.0, torch.zeros_like(hue))
    return torch.stack((hue, saturation, value), dim=0)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    hue, saturation, value = hsv.unbind(dim=0)
    sector = torch.floor(hue * 6.0).to(torch.int64) % 6
    fraction = hue * 6.0 - torch.floor(hue * 6.0)
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    options = torch.stack(
        (
            torch.stack((value, t, p)),
            torch.stack((q, value, p)),
            torch.stack((p, value, t)),
            torch.stack((p, q, value)),
            torch.stack((t, p, value)),
            torch.stack((value, p, q)),
        )
    )
    return options.gather(
        0, sector.unsqueeze(0).unsqueeze(0).expand(1, 3, *sector.shape)
    ).squeeze(0)


def _gaussian_blur(image: torch.Tensor, radius: float) -> torch.Tensor:
    sigma = max(float(radius), 1e-6)
    half_width = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(
        -half_width, half_width + 1, device=image.device, dtype=image.dtype
    )
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = image.shape[0]
    horizontal = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    batch = image.unsqueeze(0)
    batch = F.pad(batch, (half_width, half_width, 0, 0), mode="reflect")
    batch = F.conv2d(batch, horizontal, groups=channels)
    batch = F.pad(batch, (0, 0, half_width, half_width), mode="reflect")
    return F.conv2d(batch, vertical, groups=channels).squeeze(0)


def apply_appearance(
    image: torch.Tensor, appearance: AppearanceTransformRecord
) -> torch.Tensor:
    """Replay appearance only; no spatial operation is inverted later."""
    output = (image * appearance.brightness_factor).clamp(0.0, 1.0)
    contrast_mean = _rgb_to_grayscale(output).mean()
    output = (
        contrast_mean + appearance.contrast_factor * (output - contrast_mean)
    ).clamp(0.0, 1.0)
    grayscale = _rgb_to_grayscale(output)
    output = (
        grayscale + appearance.saturation_factor * (output - grayscale)
    ).clamp(0.0, 1.0)
    if output.shape[0] == 3 and appearance.hue_factor != 0.0:
        hsv = _rgb_to_hsv(output)
        hsv[0] = (hsv[0] + appearance.hue_factor) % 1.0
        output = _hsv_to_rgb(hsv).clamp(0.0, 1.0)
    if appearance.gaussian_blur:
        if appearance.gaussian_blur_radius is None:
            raise ValueError("A sampled Gaussian blur requires its stored radius.")
        output = _gaussian_blur(output, appearance.gaussian_blur_radius)
    return output


def _validate_canonical_pair(
    image: torch.Tensor, mask: torch.Tensor, contract: BWCRAugmentationContract
) -> None:
    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError("Canonical image/mask must have shapes [C,H,W] and [1,H,W].")
    if image.shape[-2:] != contract.image_size or mask.shape[-2:] != contract.image_size:
        raise ValueError(
            f"BWCR views must start from canonical {contract.image_size} tensors."
        )
    if image.shape[0] != 3:
        raise ValueError("BWCR canonical source images must contain exactly three RGB channels.")
    if mask.shape[0] != 1:
        raise ValueError("BWCR binary masks must contain exactly one channel.")
    if not torch.is_floating_point(image) or not torch.is_floating_point(mask):
        raise ValueError("Canonical image and mask tensors must be floating point.")
    if not torch.isfinite(image).all() or torch.any((image < 0) | (image > 1)):
        raise ValueError("Canonical BWCR source images must be finite and lie in [0, 1].")
    if torch.any((mask != 0) & (mask != 1)):
        raise ValueError("Canonical BWCR source masks must be binary.")


def apply_bwcr_view_record(
    canonical_image: torch.Tensor,
    canonical_mask: torch.Tensor,
    record: BWCRViewTransformRecord,
    contract: BWCRAugmentationContract,
) -> BWCRSourceView:
    """Create one view deterministically from a canonical source pair."""
    _validate_canonical_pair(canonical_image, canonical_mask, contract)
    image = apply_forward_geometry(
        canonical_image, record.geometry, interpolation="bilinear"
    )
    mask = apply_forward_geometry(
        canonical_mask, record.geometry, interpolation="nearest"
    )
    mask = (mask >= 0.5).to(canonical_mask.dtype)
    validity = apply_forward_geometry(
        torch.ones_like(canonical_mask), record.geometry, interpolation="nearest"
    )
    validity = (validity >= 0.5).to(canonical_mask.dtype)
    image = apply_appearance(image.clamp(0.0, 1.0), record.appearance)
    mean = image.new_tensor(contract.normalize_mean).view(-1, 1, 1)
    std = image.new_tensor(contract.normalize_std).view(-1, 1, 1)
    image = (image - mean) / std
    return BWCRSourceView(image, mask, validity, record)


def sample_bwcr_view(
    canonical_image: torch.Tensor,
    canonical_mask: torch.Tensor,
    contract: BWCRAugmentationContract,
    *,
    experiment_seed: int,
    epoch: int,
    dataset_name: str,
    image_id: str,
    view_index: int,
) -> BWCRSourceView:
    """Sample and apply one independently keyed source view."""
    record = sample_bwcr_view_record(
        contract,
        experiment_seed=experiment_seed,
        epoch=epoch,
        dataset_name=dataset_name,
        image_id=image_id,
        view_index=view_index,
    )
    return apply_bwcr_view_record(canonical_image, canonical_mask, record, contract)


def sample_bwcr_pair(
    canonical_image: torch.Tensor,
    canonical_mask: torch.Tensor,
    contract: BWCRAugmentationContract,
    *,
    experiment_seed: int,
    epoch: int,
    dataset_name: str,
    image_id: str,
) -> BWCRSourceViewPair:
    """Create two independent views from the same canonical source tensors."""
    common = {
        "experiment_seed": experiment_seed,
        "epoch": epoch,
        "dataset_name": dataset_name,
        "image_id": image_id,
    }
    return BWCRSourceViewPair(
        view_0=sample_bwcr_view(
            canonical_image, canonical_mask, contract, view_index=0, **common
        ),
        view_1=sample_bwcr_view(
            canonical_image, canonical_mask, contract, view_index=1, **common
        ),
    )


def inverse_bwcr_validity(view: BWCRSourceView) -> torch.Tensor:
    """Return one view's forward-valid pixels in canonical coordinates."""
    aligned = inverse_align_bwcr_tensor(
        view.forward_validity, view.record.geometry, interpolation="nearest"
    )
    return (aligned >= 0.5).to(view.forward_validity.dtype)


def bwcr_validity_intersection(pair: BWCRSourceViewPair) -> torch.Tensor:
    """Return binary canonical support valid in both independently warped views."""
    validity_0 = inverse_bwcr_validity(pair.view_0)
    validity_1 = inverse_bwcr_validity(pair.view_1)
    return (validity_0 * validity_1).clamp(0.0, 1.0)
