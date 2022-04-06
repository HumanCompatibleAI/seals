# base stage contains just binary dependencies.
# This is used in the CI build.
FROM nvidia/cuda:10.0-runtime-ubuntu18.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

RUN    apt-get update -q \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    parallel \
    patchelf \
    python3.7 \
    python3.7-dev \
    python3-pip \
    rsync \
    software-properties-common \
    unzip \
    vim \
    virtualenv \
    xpra \
    wget \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8

# Install mujoco211
RUN mkdir -p /root/.mujoco \
    && wget -P /root/.mujoco "https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz" \
    && tar -zxvf /root/.mujoco/mujoco-2.1.1-linux-x86_64.tar.gz --no-same-owner -C /root/.mujoco/ \
    && rm /root/.mujoco/mujoco-2.1.1-linux-x86_64.tar.gz

# Set the PATH to the venv before we create the venv, so it's visible in base.
# This is since we may create the venv outside of Docker, e.g. in CI
# or by binding it in for local development.
ENV PATH="/venv/bin:$PATH"

# From mujoco-py Dockerfile and Documentation https://github.com/openai/mujoco-py/tree/v2.1.2.14
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco-2.1.1/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH /root/.mujoco/mujoco-2.1.1/

# Run Xdummy mock X server by default so that rendering will work.
COPY ci/xorg.conf /etc/dummy_xorg.conf
COPY ci/Xdummy-entrypoint.py /usr/bin/Xdummy-entrypoint.py
ENTRYPOINT ["/usr/bin/Xdummy-entrypoint.py"]

# python-req stage contains Python venv, but not code.
# It is useful for development purposes: you can mount
# code from outside the Docker container.
FROM base as python-req

WORKDIR /seals
# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY ./setup.py ./setup.py
COPY ./README.md ./README.md
COPY ./src/seals/version.py ./src/seals/version.py
COPY ./ci/build_venv.sh ./ci/build_venv.sh
RUN    /seals/ci/build_venv.sh /venv \
    && rm -rf $HOME/.cache/pip

# full stage contains everything.
# Can be used for deployment and local testing.
FROM python-req as full

# Delay copying (and installing) the code until the very end
COPY . /seals
# Build a wheel then install to avoid copying whole directory (pip issue #2195)
RUN python3 setup.py sdist bdist_wheel
RUN pip install --upgrade dist/seals-*.whl

# Default entrypoints
CMD ["pytest", "-n", "auto", "-vv", "tests/"]
