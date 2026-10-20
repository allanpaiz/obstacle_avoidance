# TurtleBot4 Avoidance Demo

A TurtleBot4 in Gazebo detects a barrel with its lidar, plans a detour around
it, and drives to its goal. The run is recorded, the robot camera frames are
run through YOLO, and the result is rendered as videos.

Everything runs inside one Docker image; nothing else needs to be installed.

## Quick start

On a Linux machine with Docker:

```bash
cd gazebo_tb4_avoid
./run_avoid.sh
```

That one command builds the Docker image the first time, runs the simulation,
and writes the videos to `outputs/<timestamp>/`. The steps below explain each
part in more detail.

## 1. Install Docker

Install Docker Engine by following
<https://docs.docker.com/engine/install/>, then let your user run it without
`sudo`:

```bash
sudo usermod -aG docker "$USER"
```

Log out and back in, then check that it works:

```bash
docker run --rm hello-world
```

## 2. Get the code

Copy or clone this repository and open a terminal in its root folder (the one
that contains `run_avoid.sh`).

## 3. Build the image

```bash
docker build -t gazebo-tb4-avoid docker
```

This takes several minutes, needs network access, and uses about 5 GB of
disk. It installs Gazebo, ffmpeg, PyTorch (CPU), and Ultralytics YOLO, and
downloads the YOLO weights into the image. You only need to do this once.

You can skip this step: `./run_avoid.sh` builds the image itself if it does
not exist yet.

## 4. Run the simulation

```bash
./run_avoid.sh
```

A run takes about a minute after the image is built. The terminal shows the
progress: world generation, Gazebo start-up, recording (about 24 s of
simulation), YOLO detection, and rendering. It ends with:

```text
[avoid] Done: videos in .../outputs/<timestamp>
```

No network access is needed for a run.

If the machine has an Intel or AMD GPU, the script uses it for the camera and
lidar rendering. Otherwise it renders in software, which is slower but
works anywhere.

## 5. Watch the results

Each run writes four videos to its `outputs/<timestamp>/` folder:

- `avoidance_3x1.mp4`: overview, camera, and lidar side by side.
- `avoidance_simulation.mp4`: overview camera only.
- `avoidance_camera.mp4`: robot camera with YOLO boxes.
- `avoidance_lidar.mp4`: lidar view with the planned and driven paths.

## Other commands

Rerun YOLO and regenerate the videos of an earlier run, without Gazebo:

```bash
./run_avoid.sh --render outputs/<timestamp>
```

Rebuild the image after changing `docker/Dockerfile`:

```bash
./run_avoid.sh --build
```

Force software rendering, for example if the GPU causes errors:

```bash
SIM_GPU=0 ./run_avoid.sh
```

## Recorded data

Next to the videos, each run folder contains the raw recordings:

```text
frames/, simulation_frames/   camera and overview PNGs
camera_frames.jsonl           camera frame index
simulation_frames.jsonl       overview frame index
scans.jsonl                   lidar scans
poses.jsonl                   ground-truth robot pose
avoid_events.jsonl            controller phase changes
detections.jsonl              lidar clusters per scan while detecting
plan.json                     detected obstacle and planned path
yolo_detections.jsonl         YOLO boxes per camera frame
controller.log, summary.json  controller output and run summary
```

## Code layout

```text
run_avoid.sh            host entry point: builds the image, runs the container
docker/Dockerfile       Gazebo Harmonic, ffmpeg, Ultralytics YOLO + weights
docker/entrypoint.sh    in-container pipeline: world, Gazebo, record, YOLO, render
avoid/spec.py           every run parameter: start pose, barrel, sensors, gains
avoid/world.py          generates the Gazebo world
avoid/record.py         records sensors and pose; launches the controller
avoid/controller.py     hold, detect, plan, execute
avoid/lidar_clustering.py   DBSCAN + circle fit, as on the hardware
avoid/yolo_detect.py    offline YOLO pass over the camera frames
avoid/render.py         renders the videos (avoid/video.py drives ffmpeg)
```

The repository is mounted into the container when it runs, so code changes
in `avoid/` take effect on the next run without rebuilding the image.

## Troubleshooting

- `permission denied while trying to connect to the Docker daemon`: add your
  user to the `docker` group (step 1), then log out and back in.
- GPU or rendering errors: run `SIM_GPU=0 ./run_avoid.sh`.
- Package or image errors: rebuild with `./run_avoid.sh --build`.
- `--render` says a file is missing: pass a run folder from `outputs/` that
  completed.
