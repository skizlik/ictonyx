# Use YOUR verified image (Do not change this)
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    sudo \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# CUDA Runtime
RUN apt-get update && apt-get install -y \
    cuda-runtime-12-9 \
    libcudnn8 \
    libcublas-12-9 \
    libcufft-12-9 \
    libcurand-12-9 \
    libcusolver-12-9 \
    libcusparse-12-9 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -s /usr/bin/python3 /usr/bin/python

# --- THE STABILITY FIX ---
# 1. Upgrade pip
RUN pip install --upgrade pip

# 2. Install the "Golden Combination"
# We explicitly pin incompatible libraries to versions that play nice together.
# TensorFlow 2.15.0 is the last version before the Keras 3 transition (highly stable).
# MLflow < 2.11 still supports protobuf < 4.
RUN pip --default-timeout=120 install \
    "protobuf<4.0.0" \
    "tensorflow==2.15.0" \
    "mlflow<2.11.0" \
    "torch>=2.0.0" \
    jupyterlab \
    poetry \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    pillow \
    joblib \
    pytest \
    pytest-cov \
    hyperopt \
    shap \
    sphinx \
    sphinx-rtd-theme \
    numpydoc \
    ipywidgets \
    tqdm

# Set Environment
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_CACHE_DISABLE=1
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# Working directory
WORKDIR /home/appuser/projects

CMD ["/bin/bash"]
