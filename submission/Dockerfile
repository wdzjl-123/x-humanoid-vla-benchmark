FROM nvidia/cuda:12.4.1-base-ubuntu22.04

LABEL org.opencontainers.image.title="xhumanoid-vla-r7" \
      org.opencontainers.image.version="20260819" \
      org.opencontainers.image.description="Inference-only image for the shared R7 RGB-D ACT model"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/xhumanoid

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/xhumanoid/requirements.lock.txt /tmp/requirements.lock.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.10.0 torchvision==0.25.0 \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.lock.txt

WORKDIR /opt/xhumanoid
COPY src/xhumanoid/ /opt/xhumanoid/
COPY manifests/models.json /opt/xhumanoid/models.json
COPY scripts/run_model.sh /opt/xhumanoid/run_model.sh

RUN chmod 0755 /opt/xhumanoid/run_model.sh \
    && python3 -m compileall -q /opt/xhumanoid

ENTRYPOINT ["/opt/xhumanoid/run_model.sh"]
