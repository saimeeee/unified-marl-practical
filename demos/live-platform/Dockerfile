FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MPLBACKEND=Agg

# CityFlow is compiled from the fixed source included with the demo bundle.
# This image does not use CUDA, a remote server, or a developer's wheel cache.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY demos/live-platform/requirements-local.txt /tmp/requirements-local.txt
RUN python -m pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 cmake==3.31.6 \
    && python -m pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements-local.txt

COPY demos/live-platform/bundle/vendor/CityFlow/ /opt/CityFlow/
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/CityFlow \
    && python -c "import cityflow; print('CityFlow:', cityflow.__file__)"

WORKDIR /opt/unified-marl
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY licenses/ ./licenses/
COPY src/ ./src/
COPY demos/live-platform/ ./demos/live-platform/
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps . \
    && python -c "import torch, pandapower, unified_marl; assert not torch.cuda.is_available(); print('CPU runtime ready:', torch.__version__)" \
    && python -m pip check

EXPOSE 18765
CMD ["python", "demos/live-platform/start_local.py", "--host", "0.0.0.0", "--output", "/var/lib/unified-marl"]
