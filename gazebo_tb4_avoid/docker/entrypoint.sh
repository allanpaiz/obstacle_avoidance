#!/usr/bin/env bash
# Runs inside the container (see run_avoid.sh). With no argument: generate the
# world, run Gazebo headless, record the avoidance run, detect with YOLO, and
# render the videos into /outputs. With --render: only the last two steps.
set -Eeuo pipefail

STAGE="container startup"
FIX="rebuild the Docker image and check the error above"
on_error() {
  local code=$?
  echo "[avoid] ERROR during ${STAGE} (exit ${code}). Fix: ${FIX}." >&2
  exit "${code}"
}
trap on_error ERR

detect_and_render() {
  STAGE="YOLO detection"
  FIX="rebuild the Docker image so it contains the YOLO weights"
  echo "[avoid] YOLO: detecting obstacles in recorded camera frames"
  python3 -m avoid.yolo_detect /outputs
  STAGE="video rendering"
  FIX="check the recorded files and ffmpeg in the Docker image"
  echo "[avoid] Render: writing composite and individual videos"
  python3 -m avoid.render /outputs
}

case "${1:-}" in
--render)
  detect_and_render
  exit 0
  ;;
"") ;;
*)
  echo "[avoid] Unknown argument: $1 (expected nothing or --render)" >&2
  exit 2
  ;;
esac

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

WORLD_FILE=/tmp/tb4_avoid.sdf
STAGE="world generation"
FIX="check the robot description packages in the Docker image"
echo "[avoid] World: generating ${WORLD_FILE}"
python3 -m avoid.world --output "${WORLD_FILE}"

mkdir -p "${XDG_RUNTIME_DIR}"
chmod 0700 "${XDG_RUNTIME_DIR}"

GZ_PID=""
stop_gz() {
  [[ -n "${GZ_PID}" ]] || return 0
  kill "${GZ_PID}" 2>/dev/null || true
  (sleep 10; kill -9 "${GZ_PID}" 2>/dev/null) &
  local watchdog=$!
  wait "${GZ_PID}" 2>/dev/null || true
  kill "${watchdog}" 2>/dev/null || true
  GZ_PID=""
}
trap stop_gz EXIT
# The init process signals the whole process group (TINI_KILL_PROCESS_GROUP),
# so the running step stops too; then leave instead of moving on to the next one.
trap 'exit 130' INT TERM

if [[ "${SIM_GPU:-0}" == "1" ]]; then
  echo "[avoid] Sensor rendering: GPU (${SIM_GPU_DEVICE:-/dev/dri})"
else
  echo "[avoid] Sensor rendering: software (llvmpipe)"
fi

STAGE="Gazebo startup"
FIX="check GPU access or set SIM_GPU=0 for software rendering"
echo "[avoid] Gazebo: starting headless simulation"
gz sim -s -r --headless-rendering "${WORLD_FILE}" &
GZ_PID=$!

STAGE="Gazebo readiness"
FIX="check Gazebo startup output and sensor/plugin packages"
READY=0
for _ in $(seq 1 400); do
  if gz topic -l 2>/dev/null | grep -Fxq /model/turtlebot4_1/cmd_vel; then
    READY=1
    break
  fi
  kill -0 "${GZ_PID}" 2>/dev/null || { echo "[avoid] Gazebo exited" >&2; exit 3; }
  sleep 0.25
done
[[ "${READY}" -eq 1 ]] || {
  echo "[avoid] ERROR: Gazebo did not expose /model/turtlebot4_1/cmd_vel within 100 seconds." >&2
  exit 4
}

STAGE="sensor recording"
FIX="check the camera, lidar, pose topics, and the simulation logs above"
echo "[avoid] Record: waiting for sensor data"
set +e
python3 -m avoid.record --output /outputs
status=$?
set -e
stop_gz
[[ ${status} -eq 0 ]] || exit "${status}"
detect_and_render
