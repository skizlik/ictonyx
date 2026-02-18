# =============================================================================
# Ictonyx GPU Development Environment
# =============================================================================
# Layer strategy:
#   1. System packages & CUDA        (changes: almost never)
#   2. pip upgrade                    (changes: almost never)
#   3. Heavy ML frameworks            (changes: rarely)
#   4. Lighter Python dependencies    (changes: occasionally)
#   5. Ictonyx dependency resolution  (changes: when pyproject.toml changes)
#   6. Ictonyx source code            (changes: constantly — but rebuilds in seconds)
# =============================================================================

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---- Layer 1: System packages & CUDA runtime (changes almost never) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    sudo \
    vim \
    git \
    cuda-runtime-12-9 \
    libcudnn8 \
    libcublas-12-9 \
    libcufft-12-9 \
    libcurand-12-9 \
    libcusolver-12-9 \
    libcusparse-12-9 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# ---- Layer 2: pip upgrade (changes almost never) ----
RUN pip install --no-cache-dir --upgrade pip

# ---- Layer 3: Heavy ML frameworks (changes rarely) ----
# These are the multi-gigabyte downloads. Isolated so that changes to
# lighter dependencies or ictonyx code never trigger re-download.
#
# TensorFlow 2.15.0: last version before the Keras 3 transition.
# protobuf < 4: required for TF 2.15 + MLflow compatibility.
# MLflow < 2.11: last version supporting protobuf 3.x.
# torch: GPU-enabled PyTorch.
RUN pip install --no-cache-dir --default-timeout=120 \
    "protobuf<4.0.0" \
    "tensorflow==2.15.0" \
    "mlflow<2.11.0" \
    "torch>=2.0.0"

# ---- Layer 4: Python dependencies (changes occasionally) ----
# Everything else the project needs. Separated from Layer 3 so that
# adding or bumping a lightweight dependency doesn't re-download TF/PyTorch.
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    joblib \
    pillow \
    tqdm \
    hyperopt \
    shap \
    jupyterlab \
    ipywidgets \
    pytest \
    pytest-cov \
    sphinx \
    sphinx-rtd-theme \
    numpydoc

# ---- Layer 5: Ictonyx dependency resolution (changes when pyproject.toml changes) ----
# Copy ONLY the project metadata first. Docker caches this layer, so a
# code-only change in ictonyx/ skips dependency resolution entirely.
WORKDIR /home/appuser/projects/ictonyx
COPY pyproject.toml LICENSE README.md ./
COPY ictonyx/__init__.py ictonyx/__init__.py
RUN pip install --no-cache-dir -e ".[sklearn]" \
    && pip check || true

# ---- Layer 6: Ictonyx source code (changes constantly — rebuilds in seconds) ----
COPY . .
RUN pip install --no-cache-dir -e ".[sklearn]"

# ---- Environment ----
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_CACHE_DISABLE=1
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

WORKDIR /home/appuser/projects
CMD ["/bin/bash"]
