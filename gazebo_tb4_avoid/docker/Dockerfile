# syntax=docker/dockerfile:1

# Gazebo Harmonic for the simulation, the irobot/turtlebot4 description
# packages for the Create 3 and TurtleBot4 meshes, ffmpeg for the videos, and a
# CPU build of Ultralytics YOLO with its weights baked in, so a run needs no
# network access. The repository itself is bind-mounted at /demo by run_avoid.sh.
FROM ros:humble-ros-base-jammy

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
    && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        gz-harmonic \
        libgl1-mesa-dri \
        libglx-mesa0 \
        procps \
        python3-gz-msgs10 \
        python3-gz-transport13 \
        python3-matplotlib \
        python3-numpy \
        python3-pip \
        python3-yaml \
        ros-humble-irobot-create-description \
        ros-humble-turtlebot4-description \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Pinned to the versions the reference runs were made with, so detections and
# rendered frames do not drift with new releases.
RUN python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.1 torchvision==0.29.1 \
    && python3 -m pip install --no-cache-dir \
        numpy==1.26.4 opencv-python==4.11.0.86 ultralytics==8.4.183

# Ultralytics settings and weights live in the image: no settings file is
# created and nothing is downloaded at run time.
ENV YOLO_CONFIG_DIR=/opt/ultralytics \
    YOLO_MODEL=/opt/ultralytics/yolo26n.pt
RUN mkdir -p "${YOLO_CONFIG_DIR}" \
    && yolo settings sync=False >/dev/null \
    && python3 -c 'import os; from ultralytics import YOLO; YOLO(os.environ["YOLO_MODEL"])' \
    && test -s "${YOLO_MODEL}"

ENV LIBGL_ALWAYS_SOFTWARE=1 \
    MESA_GL_VERSION_OVERRIDE=4.5 \
    MESA_GLSL_VERSION_OVERRIDE=450 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_RUNTIME_DIR=/tmp/runtime-avoid

WORKDIR /demo
ENTRYPOINT ["/demo/docker/entrypoint.sh"]
