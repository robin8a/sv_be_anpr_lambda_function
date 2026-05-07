#!/usr/bin/env bash
# Merge a Firebase service-account JSON file into AWS Lambda environment variables.
#
# Why not use `Variables={KEY1=val1,...}` inline? Values like `private_key` contain
# commas and newlines and break that format. This script uses `--cli-input-json` instead.
#
# Prerequisites: aws cli, jq
#
# Usage:
#   export AWS_REGION=us-east-1
#   export AWS_PROFILE=your-profile   # optional
#   export FUNCTION_NAME=sv_be_anpr_lambda_function
#   ./scripts/update-lambda-env-from-firebase.sh path/to/service-account.json
#
# Note: Lambda has a combined size limit for all environment variables (check current
# AWS docs). If the update fails, use Secrets Manager or a single compact JSON variable.

set -euo pipefail

CREDENTIALS_FILE="${1:-${CREDENTIALS_FILE:-softicade-firebase-adminsdk-kl8d3-e023d9dd9e.json}}"
FUNCTION_NAME="${FUNCTION_NAME:-sv_be_anpr_lambda_function}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

AWS_ARGS=(--region "$REGION")
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_ARGS+=(--profile "$AWS_PROFILE")
fi

if ! command -v jq &>/dev/null; then
  echo "error: jq is required (install with your package manager)" >&2
  exit 1
fi

if [[ ! -f "$CREDENTIALS_FILE" ]]; then
  echo "error: credentials file not found: $CREDENTIALS_FILE" >&2
  exit 1
fi

# Reject nested JSON: Lambda env values must be strings (we map each top-level key to a string).
if ! jq -e 'type == "object" and all(.[]; type == "string")' "$CREDENTIALS_FILE" &>/dev/null; then
  echo "error: expected a flat JSON object with string values only (Firebase service account format)" >&2
  exit 1
fi

EXISTING_JSON=$(
  aws lambda get-function-configuration \
    "${AWS_ARGS[@]}" \
    --function-name "$FUNCTION_NAME" \
    --output json \
    | jq '.Environment.Variables // {}'
)

NEW_VARS=$(jq '.' "$CREDENTIALS_FILE")
MERGED_JSON=$(jq -n --argjson a "$EXISTING_JSON" --argjson b "$NEW_VARS" '$a * $b')

CLI_INPUT=$(
  jq -n \
    --arg fn "$FUNCTION_NAME" \
    --argjson vars "$MERGED_JSON" \
    '{FunctionName: $fn, Environment: {Variables: $vars}}'
)

TMP_JSON=$(mktemp)
trap 'rm -f "$TMP_JSON"' EXIT
printf '%s\n' "$CLI_INPUT" >"$TMP_JSON"

aws lambda update-function-configuration \
  "${AWS_ARGS[@]}" \
  --cli-input-json "file://$TMP_JSON" \
  --output json \
  | jq '{FunctionName, LastUpdateStatus, Environment: .Environment.Variables | keys}'

echo "Updated environment keys for Lambda: $FUNCTION_NAME (merged with existing variables)."
