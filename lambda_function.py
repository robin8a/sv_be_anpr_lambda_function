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


def _truthy_str(val: Any) -> bool:
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "on")


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

    etag_clean = (etag or "").strip('"')
    cache_key = f"{model_s3_uri}|{etag_clean}"
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


def _is_apigw_proxy_event(event: Any) -> bool:
    """REST API Gateway Lambda proxy integration includes httpMethod and body."""
    return isinstance(event, dict) and "httpMethod" in event and "body" in event


def _headers_lc(event: Dict[str, Any]) -> Dict[str, str]:
    h = event.get("headers")
    if not isinstance(h, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in h.items():
        if v is None:
            continue
        out[str(k).lower()] = str(v)
    return out


def _query_params(event: Dict[str, Any]) -> Dict[str, str]:
    q = event.get("queryStringParameters")
    if not isinstance(q, dict) or not q:
        return {}
    return {str(k): ("" if v is None else str(v)) for k, v in q.items()}


def _content_type(event: Dict[str, Any]) -> str:
    h = _headers_lc(event)
    ct = h.get("content-type", "") or ""
    return ct.split(";")[0].strip().lower()


def _decode_base64_image_field(b64: str) -> bytes:
    if not isinstance(b64, str) or not b64.strip():
        raise BadRequest("image_base64 (or content) must be a non-empty base64 string")
    s = b64.strip()
    if s.startswith("data:") and "base64," in s:
        s = s.split("base64,", 1)[1]
    try:
        return base64.b64decode(s, validate=False)
    except Exception as e:
        raise BadRequest(f"Invalid base64 image: {e}") from e


def _payload_from_json_obj(j: Dict[str, Any]) -> Dict[str, Any]:
    """Build internal payload from JSON object (direct invoke or API Gateway JSON body)."""
    if not isinstance(j, dict):
        raise BadRequest("JSON body must be an object")
    b64 = j.get("image_base64") or j.get("content")
    if not b64:
        raise BadRequest("image_base64 or content (base64-encoded image) is required")
    image_bytes = _decode_base64_image_field(b64)
    model_s3_uri = j.get("model_s3_uri")
    if not model_s3_uri or not isinstance(model_s3_uri, str):
        raise BadRequest("model_s3_uri is required")
    padding = int(j.get("padding", 10))
    gemini_model_name = str(j.get("gemini_model", GEMINI_DEFAULT_MODEL))
    debug = bool(j.get("debug", False))
    return {
        "image_bytes": image_bytes,
        "model_s3_uri": model_s3_uri.strip(),
        "padding": padding,
        "gemini_model": gemini_model_name,
        "debug": debug,
    }


def _model_uri_from_query_or_headers(event: Dict[str, Any]) -> Optional[str]:
    q = _query_params(event)
    m = q.get("model_s3_uri")
    if m:
        return m
    h = _headers_lc(event)
    return h.get("x-model-s3-uri") or h.get("x-model-uri")


def _parse_apigw_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    API Gateway REST proxy: body may be base64-encoded by API Gateway (isBase64Encoded).
    Supports:
    - application/json: { image_base64|content, model_s3_uri, ... }
    - Raw image bytes (binary media types): decoded body is image; model_s3_uri via query or X-Model-S3-Uri header.
    - Non-JSON string body: treated as raw base64 of image (defensive); model_s3_uri via query/header.
    """
    body_str = event.get("body")
    if body_str is None:
        raise BadRequest("Request body is required")
    if not isinstance(body_str, str):
        raise BadRequest("event.body must be a string")

    is_b64 = bool(event.get("isBase64Encoded"))
    ct = _content_type(event)
    q = _query_params(event)

    # Decode outer body when API Gateway base64-encodes the payload
    raw_bytes: bytes
    if is_b64:
        if not body_str:
            raise BadRequest("Request body is empty")
        try:
            raw_bytes = base64.b64decode(body_str)
        except Exception as e:
            raise BadRequest(f"Could not decode base64 body: {e}") from e
    else:
        raw_bytes = body_str.encode("utf-8") if body_str else b""

    # After outer decode: JSON object vs raw image
    if is_b64 and raw_bytes:
        looks_json = (
            "application/json" in ct
            or (len(raw_bytes) > 0 and raw_bytes[:1] in (b"{", b"["))
        )
        if looks_json:
            try:
                text = raw_bytes.decode("utf-8")
                j = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise BadRequest(f"Invalid JSON in body: {e}") from e
            if not isinstance(j, dict):
                raise BadRequest("JSON body must be an object")
            return _payload_from_json_obj(j)

    if not is_b64:
        if body_str.strip().startswith("{"):
            try:
                j = json.loads(body_str)
                if isinstance(j, dict):
                    return _payload_from_json_obj(j)
            except json.JSONDecodeError:
                pass
        if "application/json" in ct:
            if not body_str.strip():
                raise BadRequest("Request body is empty")
            try:
                j = json.loads(body_str)
            except json.JSONDecodeError as e:
                raise BadRequest(f"Invalid JSON body: {e}") from e
            if not isinstance(j, dict):
                raise BadRequest("JSON body must be an object")
            return _payload_from_json_obj(j)

    # Raw image (binary upload): use decoded bytes as image
    if is_b64 and raw_bytes and "application/json" not in ct:
        model_s3_uri = _model_uri_from_query_or_headers(event)
        if not model_s3_uri:
            raise BadRequest(
                "model_s3_uri is required for binary upload (query model_s3_uri or X-Model-S3-Uri header)"
            )
        padding = int(q.get("padding", 10))
        gemini_model = str(q.get("gemini_model") or os.environ.get("GEMINI_MODEL", GEMINI_DEFAULT_MODEL))
        debug = _truthy_str(q.get("debug")) or _truthy_str(_headers_lc(event).get("x-debug"))
        return {
            "image_bytes": raw_bytes,
            "model_s3_uri": model_s3_uri.strip(),
            "padding": padding,
            "gemini_model": gemini_model,
            "debug": debug,
        }

    # Plain base64 string in body (not isBase64Encoded)
    if body_str.strip() and not is_b64:
        image_bytes = _decode_base64_image_field(body_str)
        model_s3_uri = _model_uri_from_query_or_headers(event)
        if not model_s3_uri:
            raise BadRequest(
                "model_s3_uri is required (query model_s3_uri or X-Model-S3-Uri header) when body is raw base64"
            )
        padding = int(q.get("padding", 10))
        gemini_model = str(q.get("gemini_model") or GEMINI_DEFAULT_MODEL)
        debug = _truthy_str(q.get("debug")) or _truthy_str(_headers_lc(event).get("x-debug"))
        return {
            "image_bytes": image_bytes,
            "model_s3_uri": model_s3_uri.strip(),
            "padding": padding,
            "gemini_model": gemini_model,
            "debug": debug,
        }

    raise BadRequest("Request body is required or could not be parsed")


def _parse_direct_invoke(event: Dict[str, Any]) -> Dict[str, Any]:
    return _payload_from_json_obj(event)


def _apigw_json_response(status_code: int, body_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body_obj),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    API Gateway REST (proxy): POST body is JSON with base64 image, or binary image with model_s3_uri in query/header.
    Set GEMINI_API_KEY on the Lambda (do not pass in the request body).

    Direct invoke (no API Gateway envelope):
      { "image_base64" or "content", "model_s3_uri", optional padding, debug, gemini_model }

    Returns:
      API Gateway: { statusCode, headers, body: JSON }
      Direct: { "plate": "<string|null>", ... }
    """
    ev = event or {}
    apigw = _is_apigw_proxy_event(ev)
    payload: Dict[str, Any] = {}
    debug = False

    try:
        if apigw:
            payload = _parse_apigw_request(ev)
        else:
            payload = _parse_direct_invoke(ev if isinstance(ev, dict) else {})

        debug = bool(payload.get("debug", False))
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not gemini_api_key:
            raise BadRequest("GEMINI_API_KEY environment variable is required")

        model_s3_uri = payload["model_s3_uri"]
        padding = int(payload.get("padding", 10))
        gemini_model_name = str(payload.get("gemini_model", GEMINI_DEFAULT_MODEL))
        image_bytes = payload["image_bytes"]

        cached_model = _load_yolo_model_cached(model_s3_uri)
        im0 = _decode_image_cv2(image_bytes)

        crop_b64, xyxy = _detect_plate_crop_jpeg_b64(cached_model.model, im0, padding=padding)
        if not crop_b64:
            out: Dict[str, Any] = {"plate": None}
            if debug:
                out["debug"] = {"detected": False}
            return _apigw_json_response(200, out) if apigw else out

        plate = _gemini_extract_text(
            base64_jpeg=crop_b64,
            gemini_api_key=gemini_api_key,
            gemini_model_name=gemini_model_name,
        )

        if debug:
            dbg = {
                "detected": True,
                "box_xyxy": xyxy,
                "model_s3_uri": model_s3_uri,
                "gemini_model": gemini_model_name,
            }
            out = {"plate": plate, "debug": dbg}
            return _apigw_json_response(200, out) if apigw else out
        out = {"plate": plate}
        return _apigw_json_response(200, out) if apigw else out

    except BadRequest as e:
        logger.warning("Bad request: %s", str(e))
        err_body = {"plate": None, "error": str(e)}
        return _apigw_json_response(400, err_body) if apigw else err_body
    except Exception as e:
        logger.exception("Unhandled error: %s", str(e))
        err_body: Dict[str, Any] = {"plate": None, "error": "internal_error"}
        if debug:
            err_body["detail"] = str(e)
        return _apigw_json_response(500, err_body) if apigw else err_body

