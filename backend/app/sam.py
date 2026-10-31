"""Tighten rough LLM boxes with SAM 2.1 (box prompt -> mask -> mask bounding box)."""
from __future__ import annotations

import gc
import logging

import numpy as np
from PIL import Image

from .geometry import area, intersection, iou, pad_box

log = logging.getLogger(__name__)

MODEL_ID = "facebook/sam2.1-hiera-small"
BATCH = 16
MIN_IOU_WITH_PROMPT = 0.35


def accept_mask_box(prompt: list[float], mask: np.ndarray) -> list[int] | None:
    """Return the mask's bounding box unless it clearly latched onto something else."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    box = [x1, y1, int(xs.max()) - x1 + 1, int(ys.max()) - y1 + 1]
    if iou(prompt, box) < MIN_IOU_WITH_PROMPT:
        return None
    # The mask may only shrink the box or nudge past it, never bleed far outside.
    if intersection(prompt, box) < 0.8 * area(box):
        return None
    return box


def _infer(image: Image.Image, boxes: list[list[float]], device: str) -> list[tuple[list[int] | None, np.ndarray]]:
    import torch
    from transformers import Sam2Model, Sam2Processor

    processor = Sam2Processor.from_pretrained(MODEL_ID)
    model = Sam2Model.from_pretrained(MODEL_ID).to(device).eval()
    output: list[tuple[list[int] | None, np.ndarray]] = []
    try:
        for start in range(0, len(boxes), BATCH):
            batch = boxes[start : start + BATCH]
            xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in batch]
            inputs = processor(images=image, input_boxes=[xyxy], return_tensors="pt").to(device)
            with torch.inference_mode():
                result = model(**inputs, multimask_output=False)
            masks = processor.post_process_masks(result.pred_masks.cpu(), inputs["original_sizes"])[0]
            for prompt, mask in zip(batch, masks):
                found = mask[0].numpy() > 0
                output.append((accept_mask_box(prompt, found), found))
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return output


def _run(image: Image.Image, boxes: list[list[float]]) -> list[tuple[list[int] | None, np.ndarray]]:
    import torch

    rgb = image.convert("RGB")
    try:
        if not torch.cuda.is_available():
            raise torch.cuda.OutOfMemoryError("no cuda")
        return _infer(rgb, boxes, "cuda")
    except torch.cuda.OutOfMemoryError:
        log.warning("SAM: GPU unavailable or out of memory, falling back to CPU")
        return _infer(rgb, boxes, "cpu")


def tighten(image: Image.Image, boxes: list[list[float]]) -> tuple[list[list[float]], int]:
    """Return refined boxes and how many were refined. Raises if SAM cannot run at all."""
    if not boxes:
        return [], 0
    refined: list[list[float]] = []
    count = 0
    for prompt, (box, _) in zip(boxes, _run(image, boxes)):
        if box is None:
            refined.append(prompt)
        else:
            refined.append(pad_box(box, 0.0, 3, image.width, image.height))
            count += 1
    return refined, count


def masks(image: Image.Image, boxes: list[list[float]]) -> list[np.ndarray | None]:
    """One image-sized mask per box, or None where it latched onto something else."""
    if not boxes:
        return []
    return [found if box is not None else None for box, found in _run(image, boxes)]
