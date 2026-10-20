#!/usr/bin/env bash
# One TurtleBot4 lidar obstacle-avoidance run in the demo arena, recorded,
# detected with YOLO, and rendered as videos.
#
#   ./run_avoid.sh                record a run and render it
#   ./run_avoid.sh --render DIR   re-run YOLO and re-render an earlier capture
#   ./run_avoid.sh --build        rebuild the Docker image, then record a run
#
# The image is built automatically the first time. SIM_GPU=auto (default) uses
# an Intel/AMD GPU for sensor rendering when one is found, 1 requires one, and
# 0 always renders in software.
set -Eeuo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
    ./run_avoid.sh [--build]
    ./run_avoid.sh --render OUTPUT_DIR
EOF
}

STAGE="startup"
FIX="check Docker and the repository paths"
on_error() {
    local code=$?
    echo "[avoid] ERROR during ${STAGE} (exit ${code}). Fix: ${FIX}." >&2
    exit "${code}"
}
trap on_error ERR

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${SIM_IMAGE:-gazebo-tb4-avoid:latest}"
CONTAINER="gazebo-tb4-avoid-$$"
trap 'docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true' EXIT

# Prints "<card node> <render node>" for the first GPU Mesa can drive. Cameras
# and lidars render through Mesa's EGL, so Intel (i915/xe) and AMD
# (amdgpu/radeon) work; the proprietary NVIDIA driver and Docker Desktop
# (no /dev/dri) fall back to software rendering.
find_mesa_gpu() {
    [[ "$(uname -s)" == "Linux" ]] || return 0
    local render driver card
    for render in /sys/class/drm/renderD*; do
        [[ -e "${render}/device/driver" ]] || continue
        driver="$(basename "$(readlink -f "${render}/device/driver")")"
        case "${driver}" in
        i915 | xe | amdgpu | radeon) ;;
        *) continue ;;
        esac
        for card in "${render}"/device/drm/card*; do
            [[ -e "${card}" ]] || continue
            echo "/dev/dri/$(basename "${card}") /dev/dri/$(basename "${render}")"
            return 0
        done
    done
}

build_image() {
    STAGE="Docker image build"
    FIX="check network access and the output of docker build above"
    echo "[avoid] Image: building ${IMAGE} (the first build takes a while)"
    docker build -t "${IMAGE}" "${REPO_DIR}/docker"
}

# Runs the container with the repository at /demo and $1 at /outputs.
run_container() {
    local output="$1"
    shift
    docker run --rm --init --name "${CONTAINER}" --shm-size 2g \
        -e TINI_KILL_PROCESS_GROUP=1 \
        -v "${REPO_DIR}:/demo:ro" -v "${output}:/outputs" \
        "${GPU_ARGS[@]}" "${IMAGE}" "$@"
}

command -v docker >/dev/null 2>&1 || {
    echo "[avoid] ERROR: Docker is not installed or not on PATH." >&2
    echo "[avoid] Fix: install Docker, then retry." >&2
    exit 2
}

BUILD=0
RENDER_DIR=""
case "${1:-}" in
--render)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    RENDER_DIR="$2"
    ;;
--build)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    BUILD=1
    ;;
"") ;;
*)
    usage
    exit 2
    ;;
esac

if [[ "${BUILD}" -eq 1 ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    build_image
fi
GPU_ARGS=()

if [[ -n "${RENDER_DIR}" ]]; then
    [[ -d "${RENDER_DIR}" ]] || { echo "[avoid] ERROR: capture directory does not exist: ${RENDER_DIR}" >&2; exit 2; }
    for required in summary.json camera_frames.jsonl scans.jsonl poses.jsonl; do
        [[ -f "${RENDER_DIR}/${required}" ]] || {
            echo "[avoid] ERROR: capture is missing ${required}: ${RENDER_DIR}" >&2
            echo "[avoid] Fix: pass the output directory from a completed run." >&2
            exit 2
        }
    done
    OUTPUT_DIR="$(cd "${RENDER_DIR}" && pwd)"
    STAGE="re-rendering ${OUTPUT_DIR}"
    FIX="check the capture files and the Docker image"
    echo "[avoid] Re-rendering ${OUTPUT_DIR}"
    run_container "${OUTPUT_DIR}" --render
    exit 0
fi

GPU_NODES=""
case "${SIM_GPU:-auto}" in
auto | 1) GPU_NODES="$(find_mesa_gpu)" ;;
0) ;;
*)
    echo "[avoid] ERROR: SIM_GPU must be auto, 1, or 0 (got '${SIM_GPU}')." >&2
    exit 2
    ;;
esac
if [[ -n "${GPU_NODES}" ]]; then
    read -r GPU_CARD GPU_RENDER <<<"${GPU_NODES}"
    GPU_ARGS=(--device "${GPU_CARD}" --device "${GPU_RENDER}"
              -e LIBGL_ALWAYS_SOFTWARE=0 -e SIM_GPU=1 -e "SIM_GPU_DEVICE=${GPU_RENDER}")
    RENDERER="GPU (${GPU_RENDER})"
elif [[ "${SIM_GPU:-auto}" == "1" ]]; then
    echo "[avoid] ERROR: SIM_GPU=1 but no Intel or AMD GPU was found under /sys/class/drm." >&2
    exit 2
else
    RENDERER="software (llvmpipe)"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
STAGE="preparing the output directory"
FIX="check write permissions for ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
echo "[avoid] Output directory: ${OUTPUT_DIR}; sensor rendering: ${RENDERER}"
STAGE="simulation run"
FIX="check the Gazebo and recorder output above, or try SIM_GPU=0"
run_container "${OUTPUT_DIR}"
echo "[avoid] Done: videos in ${OUTPUT_DIR}"
