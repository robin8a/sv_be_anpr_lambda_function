# Prompts

## Lambda function

Help to create an aws lambda function with the following and name as "sv_be_anpr_lambda_function", for automatic number plate recognition

- Use the working @single-how-to-use-ultralytics-yolo-with-openai-for-number-plate-recognition.ipynb as template to create the function
- Receives a **base64-encoded image** and **`model_s3_uri`** (YOLO weights in S3). **Gemini** is configured via the **`GEMINI_API_KEY` Lambda environment variable** (do not send API keys in the request body).
- Return the plate number
- Use gemini
- Don't download the model `.pt` from the client; it is loaded from S3 using `model_s3_uri`
- Don't display the image


## Update Lambda function Base 64 (API Gateway REST)

The handler supports **API Gateway REST (proxy integration)**. Configure the Lambda with **`GEMINI_API_KEY`** (and optionally **`GEMINI_MODEL`**) in the environment.

### 1. JSON body (typical; payload under API Gateway limits)

Client sends **`Content-Type: application/json`** with:

| Field | Required | Description |
|--------|-----------|-------------|
| `image_base64` or `content` | Yes | Base64 of the image (optional `data:image/...;base64,` prefix) |
| `model_s3_uri` | Yes | `s3://bucket/path/to/model.pt` |
| `gemini_api_key` | No | Gemini API key (supported, but **prefer** using `GEMINI_API_KEY` env var) |
| `padding` | No | Crop padding (default `10`) |
| `gemini_model` | No | Overrides default; else `GEMINI_MODEL` env or `gemini-2.5-flash` |
| `debug` | No | If true, includes `debug` object in the JSON response |

Example body:

```json
{
  "image_base64": "iVBORw0KGgo...",
  "model_s3_uri": "s3://your-bucket/models/anpr-demo-model.pt",
  "gemini_api_key": "AIza....",
  "padding": 10,
  "debug": false
}
```

Lambda returns **`statusCode` 200** and **`body`** as a JSON string: `{"plate":"<string-or-null>", ...}`.

### 2. Binary upload (API Gateway binary media types)

In API Gateway, add binary media types (e.g. `image/jpeg`, `image/png`). The integration passes **`isBase64Encoded`: true** and **`body`** as base64 of the raw file.

For this mode, pass **`model_s3_uri`** as a **query parameter** or **`X-Model-S3-Uri`** header (required). Optional: `padding`, `gemini_model`, `debug` via query or `X-Debug` header.

### 3. Direct Lambda invoke (no API Gateway envelope)

Invoke with the same fields as the JSON object above (no `httpMethod` / `body` wrapper). Response is a plain object: `{"plate": ...}`.

### Errors (API Gateway)

- **400** – validation errors (`error` message).
- **500** – internal errors (`error`: `"internal_error"`; `detail` only when `debug` was true on the request).
