"""Frozen Grounding DINO and SAM2 target-category inspection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

Detection = tuple[str, float, tuple[float, float, float, float]]
Detector = Callable[[np.ndarray, tuple[str, ...]], tuple[Detection, ...]]
Segmenter = Callable[[np.ndarray, tuple[float, float, float, float]], np.ndarray]


@dataclass(frozen=True)
class InstanceDetection:
    category: str
    confidence: float
    mask: np.ndarray


def detect_instances(detector: Detector, segmenter: Segmenter,
                     rgb: np.ndarray, category: str) -> tuple[InstanceDetection, ...]:
    image = np.asarray(rgb)[..., :3]
    found = []
    for label, score, box in detector(image, (category,)):
        if label != category:
            continue
        mask = np.asarray(segmenter(image, box), dtype=bool)
        if mask.shape == image.shape[:2] and mask.any():
            found.append(InstanceDetection(label, float(score), mask))
    return tuple(found)


def detect_category(
    detector: Detector, segmenter: Segmenter, rgb: np.ndarray, category: str
) -> bool:
    """Use only the rendered RGB image and public target category for STOP."""
    return bool(detect_instances(detector, segmenter, rgb, category))


class GroundedSAMInspector:
    """Frozen public models; the private semantic image never enters these heads."""

    def __init__(self, config_path: Path, *, dino_model: str, sam_model: str) -> None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.detector, self.segmenter = _load_models(
            dino_model, sam_model, config
        )

    def __call__(self, rgb: np.ndarray, _depth: np.ndarray, category: str) -> bool:
        return detect_category(self.detector, self.segmenter, rgb, category)

    def detect_instances(self, rgb: np.ndarray, category: str) -> tuple[InstanceDetection, ...]:
        return detect_instances(self.detector, self.segmenter, rgb, category)

    def close(self) -> None:
        del self.detector, self.segmenter


def _load_models(dino_model: str, sam_model: str, config: dict):
    import torch
    from PIL import Image
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    device = config["device"]
    dino_revision = config["grounding_dino"]["revision"]
    sam_revision = config["sam"]["revision"]
    dino_kwargs = {"revision": dino_revision} if dino_model == config["grounding_dino"]["model_id"] else {}
    sam_kwargs = {"revision": sam_revision} if sam_model == config["sam"]["model_id"] else {}
    dino_processor = AutoProcessor.from_pretrained(dino_model, **dino_kwargs)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        dino_model, **dino_kwargs
    ).to(device).eval()
    sam_processor = Sam2Processor.from_pretrained(sam_model, **sam_kwargs)
    sam = Sam2Model.from_pretrained(sam_model, **sam_kwargs).to(device).eval()

    def detector(rgb: np.ndarray, categories: tuple[str, ...]) -> tuple[Detection, ...]:
        image = Image.fromarray(rgb)
        inputs = dino_processor(
            images=image, text=[list(categories)], return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            outputs = dino(**inputs)
        processed = dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=config["grounding_dino"]["box_threshold"],
            text_threshold=config["grounding_dino"]["text_threshold"],
            target_sizes=[image.size[::-1]],
        )[0]
        labels = processed.get("text_labels", processed.get("labels", ()))
        return tuple(
            (str(label), float(score), tuple(float(value) for value in box))
            for label, score, box in zip(
                labels, processed["scores"], processed["boxes"], strict=True
            )
        )

    def segmenter(rgb: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        image = Image.fromarray(rgb)
        inputs = sam_processor(
            images=image, input_boxes=[[[*box]]], return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            outputs = sam(**inputs)
        masks = sam_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        scores = outputs.iou_scores[0, 0].cpu()
        return masks[0, int(scores.argmax())].numpy().astype(bool)

    return detector, segmenter
