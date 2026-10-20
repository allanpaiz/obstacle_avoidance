#!/usr/bin/env python3
"""Write the Gazebo world for the avoidance run (see spec.py).

The demo arena (4.8 m x 3.4 m walls, centre island, route loop) with
turtlebot4_1 at the start, carrying a lidar and a forward camera; a fixed
overview camera watching the arena; and a barrel standing on the route.

  python3 -m avoid.world --output /tmp/tb4_avoid.sdf
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from . import spec as run


CREATE_MESHES = Path("/opt/ros/humble/share/irobot_create_description/meshes")
TB4_MESHES = Path("/opt/ros/humble/share/turtlebot4_description/meshes")


def f(value: float) -> str:
    return f"{float(value):.8g}"


def uri(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required robot mesh is missing: {path}")
    return f"file://{path}"


def material(color: str) -> str:
    return f"""
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
            <specular>0.08 0.08 0.08 1</specular>
          </material>"""


def mesh_visual(name: str, path: Path, pose: str, *, color: str = "0.10 0.12 0.14 1") -> str:
    return f"""
        <visual name="{name}">
          <pose>{pose}</pose>
          <geometry>
            <mesh>
              <uri>{uri(path)}</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>{material(color)}
        </visual>"""


def diagonal_inertial(mass: float, ixx: float, iyy: float, izz: float, pose: str = "0 0 0 0 0 0") -> str:
    return f"""
        <inertial>
          <pose>{pose}</pose>
          <mass>{f(mass)}</mass>
          <inertia>
            <ixx>{f(ixx)}</ixx><iyy>{f(iyy)}</iyy><izz>{f(izz)}</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
          </inertia>
        </inertial>"""


def wheel_link(name: str, y: float) -> str:
    return f"""
      <link name="{name}_wheel">
        <pose relative_to="base_link">0 {f(y)} 0.03575 -1.5708 0 0</pose>
{diagonal_inertial(0.2, 0.00009, 0.00009, 0.00013)}
        <collision name="collision">
          <geometry><cylinder><radius>0.03575</radius><length>0.015</length></cylinder></geometry>
          <surface><friction><ode><mu>20</mu><mu2>20</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>0.03575</radius><length>0.015</length></cylinder></geometry>
{material("0.025 0.025 0.025 1")}
        </visual>
      </link>"""


def base_physics_and_drive(max_linear_velocity: float) -> str:
    return f"""
{wheel_link("left", 0.1165)}
{wheel_link("right", -0.1165)}
      <link name="caster_contact">
{diagonal_inertial(0.05, 0.00002, 0.00002, 0.00002)}
        <collision name="front_caster">
          <pose>0.125 0 0.012 0 0 0</pose>
          <geometry><sphere><radius>0.012</radius></sphere></geometry>
          <surface><friction><ode><mu>0.2</mu><mu2>0.2</mu2><slip1>1</slip1><slip2>1</slip2></ode></friction></surface>
        </collision>
        <collision name="rear_caster">
          <pose>-0.125 0 0.012 0 0 0</pose>
          <geometry><sphere><radius>0.012</radius></sphere></geometry>
          <surface><friction><ode><mu>0.2</mu><mu2>0.2</mu2><slip1>1</slip1><slip2>1</slip2></ode></friction></surface>
        </collision>
      </link>

      <joint name="left_wheel_joint" type="revolute">
        <parent>base_link</parent><child>left_wheel</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
      </joint>
      <joint name="right_wheel_joint" type="revolute">
        <parent>base_link</parent><child>right_wheel</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
      </joint>
      <joint name="caster_joint" type="fixed"><parent>base_link</parent><child>caster_contact</child></joint>

      <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.233</wheel_separation>
        <wheel_radius>0.03575</wheel_radius>
        <odom_publish_frequency>40</odom_publish_frequency>
        <max_linear_acceleration>0.55</max_linear_acceleration>
        <min_linear_acceleration>-0.55</min_linear_acceleration>
        <max_angular_acceleration>1.2</max_angular_acceleration>
        <min_angular_acceleration>-1.2</min_angular_acceleration>
        <max_linear_velocity>{f(max_linear_velocity)}</max_linear_velocity>
        <min_linear_velocity>{f(-max_linear_velocity)}</min_linear_velocity>
        <max_angular_velocity>0.9</max_angular_velocity>
        <min_angular_velocity>-0.9</min_angular_velocity>
      </plugin>"""


def sensors() -> str:
    return f"""
        <sensor name="rplidar" type="gpu_lidar">
          <pose>{run.LIDAR_POSE}</pose>
          <topic>{run.LIDAR_TOPIC}</topic>
          <update_rate>{run.LIDAR_RATE}</update_rate><always_on>1</always_on>
          <lidar>
            <scan><horizontal>
              <samples>{run.LIDAR_SAMPLES}</samples><resolution>1</resolution>
              <min_angle>-3.14159265</min_angle><max_angle>3.14159265</max_angle>
            </horizontal></scan>
            <range><min>{run.LIDAR_MIN_RANGE}</min><max>{run.LIDAR_MAX_RANGE}</max><resolution>0.01</resolution></range>
          </lidar>
        </sensor>
        <sensor name="camera" type="camera">
          <pose>{run.CAMERA_POSE}</pose>
          <topic>{run.CAMERA_TOPIC}</topic>
          <update_rate>{run.CAMERA_RATE}</update_rate><always_on>1</always_on>
          <camera>
            <horizontal_fov>{run.CAMERA_HFOV}</horizontal_fov>
            <image><width>{run.CAMERA_WIDTH}</width><height>{run.CAMERA_HEIGHT}</height><format>R8G8B8</format></image>
            <clip><near>0.03</near><far>18</far></clip>
          </camera>
        </sensor>"""


def turtlebot4() -> str:
    visuals = "".join((
        mesh_visual("create3_body", CREATE_MESHES / "body_visual.dae", "0 0 0.0392 0 0 1.5708"),
        mesh_visual("create3_bumper", CREATE_MESHES / "bumper_visual.dae", "0 0 0.0392 0 0 0"),
        mesh_visual("tb4_shell", TB4_MESHES / "shell.dae", "0 0 0.0942 1.5708 0 1.5708", color="0.08 0.10 0.12 1"),
        mesh_visual("tb4_tower", TB4_MESHES / "tower_sensor_plate.dae", "0 0 0.25257 0 0 0"),
        mesh_visual("tb4_lidar", TB4_MESHES / "rplidar.dae", "-0.04 0 0.192915 0 0 1.5708"),
        mesh_visual("tb4_camera", TB4_MESHES / "oakd_pro.dae", "-0.118 0 0.14677 0 0 0"),
    ))
    return f"""
    <model name="{run.ROBOT}">
      <pose>{run.START[0]} {run.START[1]} 0 0 0 {run.START[2]}</pose>
      <self_collide>false</self_collide>
      <link name="base_link">
{diagonal_inertial(2.3, 0.016, 0.016, 0.031, "0 0 0.0742 0 0 0")}
        <collision name="body_collision">
          <pose>0 0 0.0742 0 0 0</pose>
          <geometry><cylinder><radius>0.164</radius><length>0.06</length></cylinder></geometry>
        </collision>
{visuals}
        <visual name="identity_disc">
          <pose>0 0 0.276 0 0 0</pose>
          <geometry><cylinder><radius>0.09</radius><length>0.008</length></cylinder></geometry>
{material(run.ROBOT_COLOR)}
        </visual>
{sensors()}
      </link>
{base_physics_and_drive(0.34)}
    </model>"""


def arena() -> str:
    walls = []
    for name, pose, size in (
        ("north", "0 1.65 0.15 0 0 0", "4.8 0.08 0.30"),
        ("south", "0 -1.65 0.15 0 0 0", "4.8 0.08 0.30"),
        ("east", "2.36 0 0.15 0 0 0", "0.08 3.38 0.30"),
        ("west", "-2.36 0 0.15 0 0 0", "0.08 3.38 0.30"),
    ):
        walls.append(f'<collision name="{name}_collision"><pose>{pose}</pose><geometry><box><size>{size}</size></box></geometry></collision>')
        walls.append(f'<visual name="{name}_visual"><pose>{pose}</pose><geometry><box><size>{size}</size></box></geometry>{material("0.52 0.58 0.64 1")}</visual>')

    path_segments = []
    loop = ((-1.5, 0.9), (1.5, 0.9), (1.5, -0.9), (-1.5, -0.9), (-1.5, 0.9))
    for idx, (start, end) in enumerate(zip(loop, loop[1:]), start=1):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        cx, cy = 0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])
        path_segments.append(
            f'<visual name="route_{idx}"><pose>{f(cx)} {f(cy)} 0.012 0 0 {f(math.atan2(dy, dx))}</pose>'
            f'<transparency>0.58</transparency><geometry><box><size>{f(length)} 0.035 0.004</size></box></geometry>'
            f'{material("0.12 0.48 0.72 0.42")}</visual>'
        )

    return f"""
    <model name="arena">
      <static>true</static>
      <link name="ground">
        <collision name="ground_collision"><pose>0 0 -0.01 0 0 0</pose><geometry><box><size>4.8 3.4 0.02</size></box></geometry></collision>
        <visual name="ground_visual"><pose>0 0 -0.01 0 0 0</pose><geometry><box><size>4.8 3.4 0.02</size></box></geometry>{material("0.72 0.75 0.78 1")}</visual>
        {''.join(walls)}
        {''.join(path_segments)}
        <collision name="center_island_collision"><pose>0 0 0.12 0 0 0</pose><geometry><cylinder><radius>0.38</radius><length>0.24</length></cylinder></geometry></collision>
        <visual name="center_island_visual"><pose>0 0 0.12 0 0 0</pose><geometry><cylinder><radius>0.38</radius><length>0.24</length></cylinder></geometry>{material("0.33 0.38 0.44 1")}</visual>
      </link>
    </model>"""


def overview_camera() -> str:
    return f"""
    <model name="overview_camera">
      <static>true</static>
      <pose>{run.OVERVIEW_POSE}</pose>
      <link name="camera_link">
        <sensor name="overview" type="camera">
          <topic>{run.OVERVIEW_TOPIC}</topic>
          <update_rate>{run.OVERVIEW_RATE}</update_rate><always_on>1</always_on>
          <camera>
            <horizontal_fov>{run.OVERVIEW_HFOV}</horizontal_fov>
            <image><width>{run.OVERVIEW_WIDTH}</width><height>{run.OVERVIEW_HEIGHT}</height><format>R8G8B8</format></image>
            <clip><near>0.1</near><far>30</far></clip>
          </camera>
        </sensor>
      </link>
    </model>"""


def barrel() -> str:
    """An orange safety barrel with two white bands; static, so the lidar sees a fixed arc."""
    r, h = run.BARREL_RADIUS, run.BARREL_HEIGHT
    bands = "".join(
        f'<visual name="band_{i}"><pose>0 0 {z:.3f} 0 0 0</pose>'
        f'<geometry><cylinder><radius>{r + 0.002:.3f}</radius><length>0.045</length></cylinder></geometry>'
        f'{material("0.96 0.96 0.94 1")}</visual>'
        for i, z in enumerate((h * 0.55, h * 0.78)))
    return f"""
    <model name="{run.BARREL_NAME}">
      <static>true</static>
      <pose>{run.BARREL_XY[0]} {run.BARREL_XY[1]} 0 0 0 0</pose>
      <link name="body">
        <collision name="collision"><pose>0 0 {h / 2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry></collision>
        <visual name="visual"><pose>0 0 {h / 2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
          {material("0.95 0.40 0.06 1")}</visual>
        <visual name="lid"><pose>0 0 {h + 0.008:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>{r - 0.02:.3f}</radius><length>0.016</length></cylinder></geometry>
          {material("0.20 0.20 0.22 1")}</visual>
        {bands}
      </link>
    </model>"""


def world_sdf() -> str:
    sdf = f"""<sdf version="1.9">
  <world name="{run.WORLD_NAME}">
    <physics name="1ms" type="ignored"><max_step_size>0.002</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <scene><ambient>0.72 0.74 0.76 1</ambient><background>0.88 0.90 0.92 1</background><grid>false</grid></scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse><specular>0.4 0.4 0.4 1</specular>
      <attenuation><range>1000</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
      <direction>-0.45 0.25 -0.86</direction>
    </light>
{arena()}
{turtlebot4()}
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
{overview_camera()}
{barrel()}
  </world>
</sdf>"""
    root = ET.fromstring(sdf)
    ET.indent(root, space="  ")
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(world_sdf(), encoding="utf-8")
    print(f"[avoid-world] Generated {args.output}: {run.ROBOT} at {run.START[:2]}, "
          f"barrel at {run.BARREL_XY}", flush=True)


if __name__ == "__main__":
    main()
