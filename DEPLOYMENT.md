# Deploy `sv_be_anpr_lambda_function`

This deployment flow matches the same **ECR → Lambda container image** process used in [`ts_biomass_ndvi_lambda`](https://raw.githubusercontent.com/robin8a/ts_biomass_ndvi_lambda/main/DEPLOYMENT.md).

## What this Lambda does
- Downloads **input image** from S3 (`image_s3_uri`)
- Downloads **YOLO `.pt` model** from S3 (`model_s3_uri`) into `/tmp` (cached on warm starts)
- Detects the plate with Ultralytics YOLO, crops the plate region
- Sends the cropped plate image to **Gemini** using your `gemini_api_key`
- Returns `{ "plate": "<string-or-null>" }`

Handler: `lambda_function.lambda_handler`

---

## Prereqs
- AWS CLI installed
- Docker installed
- Logged into AWS (SSO/profile)
- IAM role for Lambda execution (example below)

### Required IAM permissions (Lambda execution role)
At minimum, the Lambda execution role should allow:
- **CloudWatch Logs**: `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`
- **S3** (read model + image): `s3:GetObject`, `s3:HeadObject` on the buckets/keys you will use

---

## Variables to set (recommended)
Replace these with your values:

```sh
export AWS_PROFILE="treegacy_amplify_assets"
export AWS_REGION="us-east-1"
export ACCOUNT_ID="826331271346"

export FUNCTION_NAME="sv_be_anpr_lambda_function"
export ECR_REPO_NAME="sv_be_anpr_lambda_function_repo"
export IMAGE_NAME="sv_be_anpr_lambda_function_image"
export IMAGE_TAG="latest"
export IMAGE_NAME="sv_be_anpr_lambda_function_image"
export YOUR_LAMBDA_EXECUTION_ROLE="sv-be-ai-models-lambda-exe-role"

# arn:aws:iam::826331271346:role/sv-be-ai-models-lambda-exe-role

export LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${YOUR_LAMBDA_EXECUTION_ROLE}"
```

Login (if using SSO):

```sh
aws sso login --profile "$AWS_PROFILE"
```

---

## Create ECR repository (one time)
```sh
aws ecr create-repository \
  --repository-name "$ECR_REPO_NAME" \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE"
```

```json
{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:826331271346:repository/sv_be_anpr_lambda_function_repo",
        "registryId": "826331271346",
        "repositoryName": "sv_be_anpr_lambda_function_repo",
        "repositoryUri": "826331271346.dkr.ecr.us-east-1.amazonaws.com/sv_be_anpr_lambda_function_repo",
        "createdAt": "2026-02-16T21:39:43.203000-05:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
```

Get the repository URI (save it):

```sh
export ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
echo "$ECR_URI"
```

---

## Docker build and push to ECR
Login Docker to ECR (use `sudo` if you use `sudo docker` for build/push, so credentials are stored for the same user):

```sh
aws ecr get-login-password \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE" \
| sudo docker login \
  --username AWS \
  --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
```

Build (Lambda needs a single-platform image in Docker v2 manifest format; disable attestations and set platform):

```sh
sudo docker build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" .
```

Tag:

```sh
sudo docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
```

Push:

```sh
sudo docker push "${ECR_URI}:${IMAGE_TAG}"
```

---

## Create Lambda function (first time)
```sh
aws lambda create-function \
  --function-name "$FUNCTION_NAME" \
  --package-type Image \
  --code ImageUri="${ECR_URI}:${IMAGE_TAG}" \
  --role "$LAMBDA_ROLE_ARN" \
  --timeout 120 \
  --memory-size 2048 \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE"
```

Recommended initial settings:
- **timeout**: 120s (YOLO + Gemini call)
- **memory**: 2048–4096MB (more memory = more CPU)

---

```json
{
    "FunctionName": "sv_be_anpr_lambda_function",
    "FunctionArn": "arn:aws:lambda:us-east-1:826331271346:function:sv_be_anpr_lambda_function",
    "Role": "arn:aws:iam::826331271346:role/sv-be-ai-models-lambda-exe-role",
    "CodeSize": 0,
    "Description": "",
    "Timeout": 120,
    "MemorySize": 2048,
    "LastModified": "2026-02-17T03:29:57.627+0000",
    "CodeSha256": "e031dd49078aaa50e9ad0f3b97728042c783d0aa049093076fe6b243af94b5c7",
    "Version": "$LATEST",
    "TracingConfig": {
        "Mode": "PassThrough"
    },
    "RevisionId": "99ec16ea-e1b3-416a-9044-ae3c4419eba7",
    "State": "Pending",
    "StateReason": "The function is being created.",
    "StateReasonCode": "Creating",
    "PackageType": "Image",
    "Architectures": [
        "x86_64"
    ],
    "EphemeralStorage": {
        "Size": 512
    },
    "SnapStart": {
        "ApplyOn": "None",
        "OptimizationStatus": "Off"
    },
    "LoggingConfig": {
        "LogFormat": "Text",
        "LogGroup": "/aws/lambda/sv_be_anpr_lambda_function"
    }
}
```

## Update Lambda function (new image)
When you push a new image tag (or re-push `latest`), update the Lambda code:

```sh
aws lambda update-function-code \
  --function-name "$FUNCTION_NAME" \
  --image-uri "${ECR_URI}:${IMAGE_TAG}" \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE"
```

Update timeout/memory if needed:

```sh
aws lambda update-function-configuration \
  --function-name "$FUNCTION_NAME" \
  --timeout 120 \
  --memory-size 4096 \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE"
```

---

## Test Lambda function (direct invoke)
Edit [`EventTest.json`](./EventTest.json) with your:
- `image_s3_uri`
- `model_s3_uri`
- `gemini_api_key`

Invoke:

```sh
aws lambda invoke \
  --function-name "$FUNCTION_NAME" \
  --cli-binary-format raw-in-base64-out \
  --payload file://EventTest.json \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE" \
  response.json
```

View response:

```sh
cat response.json
```

Expected shape:
```json
{ "plate": "ABC123" }
```

Tip: `EventTest.json` includes `"debug": true` to also return `box_xyxy` and other metadata.

---

## View logs
```sh
aws logs tail "/aws/lambda/${FUNCTION_NAME}" --follow \
  --region "$AWS_REGION" \
  --profile "$AWS_PROFILE"
```

---

## Notes
- “Don’t download the model” is interpreted as **don’t download from the internet**. The `.pt` is read from **S3** and cached under `/tmp` per warm Lambda container.
- If you need a different Gemini model, pass `"gemini_model": "gemini-2.5-flash"` (or set env var `GEMINI_MODEL` in Lambda configuration).

### "Image manifest, config or layer media type ... is not supported"
Either the image was built for the wrong platform (e.g. ARM), or Docker Buildx added attestations Lambda doesn’t support. Rebuild with `--platform linux/amd64 --provenance=false --sbom=false` (see Build step above), then re-tag, push, and create/update the function again.

