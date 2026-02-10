import base64
import hashlib
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import boto3
import numpy as np


logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

S3 = boto3.client("s3")

# Warm-cache of loaded YOLO models (keyed by model_s3_uri + etag)
_YOLO_CACHE: Dict[str, "CachedYoloModel"] = {}


GEMINI_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Prompt copied from the notebook template (with small clarifications kept minimal)
GEMINI_PROMPT = """
Can you extract the vehicle number plate text inside the image?
If you are not able to extract text, please respond with None.
Only output text, please.
If any text character is not from the English language, replace it with a dot (.)
""".strip()


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


@dataclass
class CachedYoloModel:
    model_s3_uri: str
    etag: Optional[str]
    local_path: str
    model: Any  # ultralytics.YOLO


class BadRequest(ValueError):
    pass


def _parse_s3_uri(uri: str) -> S3Uri:
    if not isinstance(uri, str) or not uri:
        raise BadRequest("S3 URI must be a non-empty string")
    if not uri.startswith("s3://"):
        raise BadRequest("S3 URI must start with s3://")
    parts = uri[5:].split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise BadRequest("S3 URI must be in the form s3://bucket/key")
    return S3Uri(bucket=parts[0], key=parts[1])


def _tmp_path_for_s3(uri: S3Uri, etag: Optional[str], suffix: str) -> str:
    # Keep path deterministic and safe for /tmp
    etag_clean = (etag or "").strip('"')
    h = hashlib.sha256(f"s3://{uri.bucket}/{uri.key}|{etag_clean}".encode("utf-8")).hexdigest()[:32]
    base = os.path.basename(uri.key) or "file"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return f"/tmp/{h}__{base}{suffix}"


def _head_object(uri: S3Uri) -> Dict[str, Any]:
    return S3.head_object(Bucket=uri.bucket, Key=uri.key)


def _download_s3_to_tmp(uri: S3Uri, *, suffix: str = "") -> Tuple[str, Optional[str]]:
    head = _head_object(uri)
    etag = head.get("ETag")
    local_path = _tmp_path_for_s3(uri, etag, suffix=suffix)
    if os.path.exists(local_path):
        return local_path, etag
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    logger.info("Downloading s3://%s/%s to %s", uri.bucket, uri.key, local_path)
    S3.download_file(uri.bucket, uri.key, local_path)
    return local_path, etag


def _download_s3_bytes(uri: S3Uri) -> bytes:
    resp = S3.get_object(Bucket=uri.bucket, Key=uri.key)
    return resp["Body"].read()


def _decode_image_cv2(image_bytes: bytes) -> "np.ndarray":
    import cv2  # local import to reduce cold-start import cost slightly

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    im0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im0 is None:
        raise BadRequest("Could not decode image bytes (cv2.imdecode returned None)")
    return im0


def _load_yolo_model_cached(model_s3_uri: str) -> CachedYoloModel:
    from ultralytics import YOLO

    uri = _parse_s3_uri(model_s3_uri)
    local_path, etag = _download_s3_to_tmp(uri, suffix="")  # preserve .pt name already in key

    cache_key = f"{model_s3_uri}|{(etag or '').strip('\"')}"
    cached = _YOLO_CACHE.get(cache_key)
    if cached is not None and cached.local_path == local_path:
        return cached

    logger.info("Loading YOLO model from %s", local_path)
    model = YOLO(local_path)
    cached = CachedYoloModel(model_s3_uri=model_s3_uri, etag=etag, local_path=local_path, model=model)
    _YOLO_CACHE.clear()  # keep memory bounded (only 1 model cached by default)
    _YOLO_CACHE[cache_key] = cached
    return cached


def _detect_plate_crop_jpeg_b64(
    yolo_model: Any,
    im0: "np.ndarray",
    *,
    padding: int = 10,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    """
    Returns (base64_jpeg, xyxy) or (None, None) if no plate is detected.
    Mirrors the notebook approach: model.predict(im0)[0].boxes, crop with padding, cv2.imencode('.jpg').
    """
    import cv2

    results = yolo_model.predict(im0, verbose=False)
    if not results:
        return None, None

    boxes_obj = getattr(results[0], "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return None, None

    xyxy = boxes_obj.xyxy.cpu().numpy()
    conf = getattr(boxes_obj, "conf", None)
    if conf is not None and len(conf) == len(xyxy):
        conf_np = conf.cpu().numpy()
        best_i = int(np.argmax(conf_np))
    else:
        best_i = 0

    x1, y1, x2, y2 = xyxy[best_i].astype(int).tolist()
    h, w = im0.shape[:2]
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, w)
    y2 = min(y2 + padding, h)

    cropped = im0[y1:y2, x1:x2]
    if cropped.size == 0:
        return None, None

    ok, buf = cv2.imencode(".jpg", cropped)
    if not ok:
        return None, None

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return b64, (x1, y1, x2, y2)


def _gemini_extract_text(
    *,
    base64_jpeg: str,
    gemini_api_key: str,
    gemini_model_name: str = GEMINI_DEFAULT_MODEL,
    prompt: str = GEMINI_PROMPT,
) -> Optional[str]:
    # Notebook template uses google.generativeai + PIL.Image
    import google.generativeai as genai
    from PIL import Image

    if not gemini_api_key:
        raise BadRequest("gemini_api_key is required")

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(gemini_model_name)

    try:
        image_bytes = base64.b64decode(base64_jpeg)
        img = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content([prompt, img])
        extracted = (response.text or "").strip()
    except Exception as e:
        logger.exception("Error during Gemini text extraction: %s", str(e))
        return None

    if not extracted or extracted.lower() == "none":
        return None

    # Enforce "non-English" -> "." using ASCII as a pragmatic proxy.
    extracted = "".join(ch if ch.isascii() else "." for ch in extracted)
    extracted = extracted.strip()

    # Optional normalization that tends to help plate outputs
    extracted = re.sub(r"\s+", "", extracted).upper()

    if not extracted or extracted.lower() == "none":
        return None
    return extracted


def _coerce_event(event: Dict[str, Any]) -> Dict[str, Any]:
    # Support API Gateway proxy shape as a convenience, but direct invoke is primary.
    if isinstance(event, dict) and "body" in event and isinstance(event["body"], str):
        try:
            return json.loads(event["body"])
        except Exception:
            return event
    return event


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Expected direct-invoke event:
      {
        "image_s3_uri": "s3://bucket/path/to/image.jpg",
        "model_s3_uri": "s3://bucket/path/to/anpr-demo-model.pt",
        "gemini_api_key": "..."
      }
    Returns:
      { "plate": "<string-or-null>" }
    """
    event = _coerce_event(event or {})

    try:
        image_s3_uri = event.get("image_s3_uri")
        model_s3_uri = event.get("model_s3_uri")
        gemini_api_key = event.get("gemini_api_key")

        if not image_s3_uri:
            raise BadRequest("image_s3_uri is required")
        if not model_s3_uri:
            raise BadRequest("model_s3_uri is required")
        if not gemini_api_key:
            raise BadRequest("gemini_api_key is required")

        padding = int(event.get("padding", 10))
        gemini_model_name = str(event.get("gemini_model", GEMINI_DEFAULT_MODEL))
        debug = bool(event.get("debug", False))

        # Load model from S3 (cached on warm starts)
        cached_model = _load_yolo_model_cached(model_s3_uri)

        # Download and decode image from S3
        img_uri = _parse_s3_uri(image_s3_uri)
        img_bytes = _download_s3_bytes(img_uri)
        im0 = _decode_image_cv2(img_bytes)

        # Detect + crop plate
        crop_b64, xyxy = _detect_plate_crop_jpeg_b64(cached_model.model, im0, padding=padding)
        if not crop_b64:
            return {"plate": None, **({"debug": {"detected": False}} if debug else {})}

        plate = _gemini_extract_text(
            base64_jpeg=crop_b64,
            gemini_api_key=gemini_api_key,
            gemini_model_name=gemini_model_name,
        )

        if debug:
            return {
                "plate": plate,
                "debug": {
                    "detected": True,
                    "box_xyxy": xyxy,
                    "model_s3_uri": model_s3_uri,
                    "image_s3_uri": image_s3_uri,
                    "gemini_model": gemini_model_name,
                },
            }
        return {"plate": plate}

    except BadRequest as e:
        # Keep error shape simple for direct invoke
        logger.warning("Bad request: %s", str(e))
        return {"plate": None, "error": str(e)}
    except Exception as e:
        logger.exception("Unhandled error: %s", str(e))
        return {"plate": None, "error": "internal_error"}

