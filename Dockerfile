FROM public.ecr.aws/lambda/python:3.11

# Keep logs unbuffered
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR ${LAMBDA_TASK_ROOT}

# Compiler for packages that build from source (opencv etc.); numpy stays on wheel.
# libGL.so.1 for OpenCV/Ultralytics (AL2: libglvnd-glx; also mesa-libGL for compatibility).
#
# Also install Node.js (Firebase client SDK requires Node >= 20).
RUN yum install -y gcc gcc-c++ libglvnd-glx mesa-libGL curl && \
    curl -fsSL https://rpm.nodesource.com/setup_20.x | bash - && \
    yum install -y nodejs && \
    yum clean all

# Constrain numpy to 1.26.4 so no package pulls NumPy 2.x (build needs GCC >= 9.3)
COPY requirements.txt constraints.txt ./
# Install from wheels to avoid source builds (numpy/scipy/contourpy need newer toolchain)
RUN pip install --only-binary numpy "numpy==1.26.4" && \
    pip install --only-binary scipy "scipy>=1.11,<1.17" && \
    pip install --only-binary contourpy "contourpy>=1.2" && \
    pip install meson-python ninja pyproject-metadata setuptools wheel && \
    pip install --no-build-isolation -r requirements.txt -c constraints.txt

# Install Node deps for Firebase downloader (kept small; no dev deps).
COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

# Copy Lambda handler
COPY lambda_function.py ./
COPY firebase_download.mjs ./

# Lambda handler entrypoint
CMD ["lambda_function.lambda_handler"]

