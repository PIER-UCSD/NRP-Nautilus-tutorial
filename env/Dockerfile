# The Shi Lab image: build it once, push it, and every student's Job runs the
# exact same environment — no pip install at job start, so no network flakes,
# no half-installed requirements, no silent version drift.
#
#   bash env/build_image.sh            # build + push + print the digest
#
# The base image already contains torch + CUDA + cuDNN. Do NOT pip-upgrade
# torch here: that is what breaks working pods.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# System packages that runtime images lack and native builds need (see README step 9).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Some images ship only `python3`; make `python` work too (README step 9).
RUN ln -sf "$(command -v python3)" /usr/local/bin/python

COPY requirements-lab.txt /tmp/requirements-lab.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lab.txt

# Record what is actually inside the image, so any run can be reproduced.
RUN pip freeze > /opt/lab-env-freeze.txt
ENV LAB_ENV_FREEZE=/opt/lab-env-freeze.txt

WORKDIR /workspace
