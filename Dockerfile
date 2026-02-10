FROM public.ecr.aws/lambda/python:3.11

# Keep logs unbuffered
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR ${LAMBDA_TASK_ROOT}

# Install Python dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy Lambda handler
COPY lambda_function.py ./

# Lambda handler entrypoint
CMD ["lambda_function.lambda_handler"]

