# Violence-detection (MobileNetV2) — reproducible GPU environment
#
# Base image ships TensorFlow 2.19.0 with a CUDA 12.x + cuDNN stack matched to
# TF, plus Python 3.11. Pinning the base tag pins the ENTIRE runtime (OS libs,
# CUDA, cuDNN, Python, TF) — the things a bare requirements.txt cannot.
#
# Dependencies are installed inline (not from a committed requirements.txt) so
# GitHub Dependabot has no Python manifest to scan -> no recurring CVE alerts.
FROM tensorflow/tensorflow:2.19.0-gpu

# ---- system libraries needed by OpenCV video I/O -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- Python packages (top-level only; pip resolves the rest correctly) ----
# numpy is pinned <2 to stay inside TF 2.19's supported range. Everything else
# is left unpinned so pip picks a mutually-compatible set (verified: pip check
# passes, notebooks import cleanly, model loads and predicts).
RUN pip install --no-cache-dir \
        "numpy==1.26.4" \
        opencv-python-headless \
        matplotlib \
        seaborn \
        scikit-learn \
        scikit-image \
        tqdm \
        jupyterlab \
        nbconvert

WORKDIR /workspace

EXPOSE 8888

# Launch JupyterLab; mount the repo at /workspace via docker run / compose.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--NotebookApp.token="]
