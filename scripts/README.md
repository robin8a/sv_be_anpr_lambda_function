# Scripts

Utility scripts for deploying and operating this Lambda. Run them from the repository root unless noted.

## Prerequisites

- **AWS CLI** v2 (configured with credentials or SSO; see [DEPLOYMENT.md](../DEPLOYMENT.md))
- **`jq`** (JSON processor; install via your OS package manager)

---

## `update-lambda-env-from-firebase.sh`

Uploads every top-level key from a **Firebase service account JSON** file into the Lambda’s **environment variables**, using a merge with the function’s existing variables.

The Lambda handler initializes Firebase Admin using either env var `FIREBASE_SERVICE_ACCOUNT` (full JSON string) **or** those same keys when each is set separately, as this script does.

### Why this script exists

A naive `aws lambda update-function-configuration` call with inline `--environment "Variables={KEY=val,...}"` is unsafe for Firebase credentials: fields like `private_key` contain **commas and newlines**, which break that syntax.

This script builds a proper **`--cli-input-json`** payload so values are encoded correctly.

### Behavior

1. Reads the service account JSON (flat object; all values must be strings, as in the standard Firebase download).
2. Calls **`get-function-configuration`** and takes `Environment.Variables` (or `{}` if missing).
3. **Merges** with the JSON file: keys from the file **overwrite** the same keys on Lambda; all other existing variables are **kept**.

Updating Lambda environment replaces the whole `Environment` block in one call, but this script **re-applies** your previous variables plus the Firebase keys so you do not lose unrelated settings (for example model URI or API keys).

### Environment variables (for the script)

| Variable | Required | Description |
|----------|----------|-------------|
| `FUNCTION_NAME` | No | Lambda function name. Default: `sv_be_anpr_lambda_function`. |
| `AWS_REGION` or `AWS_DEFAULT_REGION` | No | AWS region. Default: `us-east-1`. |
| `AWS_PROFILE` | No | AWS CLI named profile (optional). |
| `CREDENTIALS_FILE` | No | Default path to the JSON file if you do not pass it as the first argument. Default: `softicade-firebase-adminsdk-kl8d3-e023d9dd9e.json`. |

The script also accepts the credentials path as **positional argument 1**, which overrides `CREDENTIALS_FILE`.

### Usage

```bash
export AWS_REGION=us-east-1
export AWS_PROFILE=your-profile    # optional
export FUNCTION_NAME=sv_be_anpr_lambda_function

./scripts/update-lambda-env-from-firebase.sh path/to/your-firebase-adminsdk.json
```

Example using defaults for function name and file name (run from repo root, with the JSON file present):

```bash
export AWS_REGION=us-east-1
./scripts/update-lambda-env-from-firebase.sh
```

Ensure your IAM principal can call:

- `lambda:GetFunctionConfiguration`
- `lambda:UpdateFunctionConfiguration`

### Output

The script prints a short JSON summary (function name, last update status, and **list of environment variable keys**) and confirms success. It does **not** print secret values.

### Limits and troubleshooting

- **Lambda environment size**: AWS applies a **combined** size limit for all environment variables. A full service account JSON split across many keys, plus other app variables, can exceed the limit. If the update fails, consider [AWS Secrets Manager](https://docs.aws.amazon.com/secretsmanager/) or [SSM Parameter Store](https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html), or store **one** variable containing the entire JSON (or base64-encoded JSON).

### Security

- Treat the Firebase JSON like a **private key**: keep it out of git (see `.gitignore`) and restrict who can run this script or read the file.
- Anyone with `lambda:GetFunctionConfiguration` can see environment variable **values** in the API response; prefer secrets services for production if policy requires it.

For the full deployment flow (ECR, image, role), see [DEPLOYMENT.md](../DEPLOYMENT.md).
