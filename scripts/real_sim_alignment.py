"""Conversion between real SO-101 joint readings and MuJoCo joint angles.

LeRobot reports joints in degrees, with 0 at the middle of the range recorded
during `lerobot-calibrate`. The MuJoCo model uses radians, with 0 at the middle
of the CAD range. Those two middles differ slightly on a home-built arm, so a
small constant offset per joint is fitted by scripts/table_touch_calibration.py:

    sim_deg = SIGN * (real_deg + offset_deg)
    real_deg = SIGN * sim_deg - offset_deg

The offsets are only valid for the calibration they were fitted on, so the file
also stores a fingerprint of that calibration.
"""

import hashlib
from pathlib import Path

import draccus
import mujoco
import numpy as np
import yaml

from lerobot.robots.so_follower import SOFollowerRobotConfig

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# +1 if the real joint and the model joint turn the same way, -1 if opposite.
# Confirmed all +1 with the digital-twin test on 2026-11-09.
SIGN = {joint: 1 for joint in ARM_JOINTS}

OFFSETS_PATH = Path("config/sim_joint_offsets.yaml")


def load_robot_config(path):
    """Read the robot section of the shared YAML config into LeRobot's typed config."""
    with open(path) as f:
        raw = draccus.load(dict, f)["robot"]
    raw.pop("type")
    return draccus.decode(SOFollowerRobotConfig, raw)


def calibration_fingerprint(calibration_fpath):
    """Short hash of the calibration file, so stale offsets can be detected."""
    return hashlib.sha256(Path(calibration_fpath).read_bytes()).hexdigest()[:12]


def load_offsets(calibration_fpath, path=OFFSETS_PATH):
    """Per-joint offsets in degrees. Zero for any joint without one, or if there is no file."""
    offsets = {joint: 0.0 for joint in ARM_JOINTS}
    if not Path(path).exists():
        return offsets
    saved = yaml.safe_load(Path(path).read_text())
    if saved.get("calibration_fingerprint") != calibration_fingerprint(calibration_fpath):
        print(f"WARNING: {path} was fitted on a different calibration. Ignoring it; refit the offsets.")
        return offsets
    offsets.update(saved["offsets_deg"])
    return offsets


def save_offsets(offsets, calibration_fpath, notes, path=OFFSETS_PATH):
    doc = {
        "calibration_fingerprint": calibration_fingerprint(calibration_fpath),
        "offsets_deg": {joint: round(float(value), 2) for joint, value in offsets.items()},
        "notes": notes,
    }
    header = "# Written by scripts/table_touch_calibration.py fit --write. sim_deg = real_deg + offset_deg\n"
    Path(path).write_text(header + yaml.safe_dump(doc, sort_keys=False))


def real_to_sim(model, reading, offsets):
    """LeRobot reading (degrees, gripper 0-100) -> MuJoCo joint angles (radians)."""
    q = {joint: SIGN[joint] * np.radians(reading[joint] + offsets[joint]) for joint in ARM_JOINTS}
    # Gripper: 0 (closed) to 100 (open) mapped linearly onto the model's range. Not yet verified.
    lo, hi = model.jnt_range[model.joint("gripper").id]
    q["gripper"] = lo + reading["gripper"] / 100 * (hi - lo)
    return q


def sim_to_real(q_sim_rad, offsets):
    """MuJoCo arm joint angles (radians, dict) -> LeRobot goal positions (degrees)."""
    return {joint: SIGN[joint] * np.degrees(q_sim_rad[joint]) - offsets[joint] for joint in ARM_JOINTS}


def sim_gripper_to_real(model, q_gripper_rad):
    """Inverse of the gripper mapping in real_to_sim: model radians -> LeRobot 0-100. Not yet verified."""
    lo, hi = model.jnt_range[model.joint("gripper").id]
    return float(np.clip((q_gripper_rad - lo) / (hi - lo) * 100, 0, 100))


def set_pose(model, data, q):
    """Write joint angles into data and update positions (no physics)."""
    for joint, angle in q.items():
        data.qpos[model.jnt_qposadr[model.joint(joint).id]] = angle
    mujoco.mj_kinematics(model, data)


class LowestPoint:
    """Lowest world height of a body's mesh vertices: where it would touch a flat surface."""

    def __init__(self, model, body_names):
        self.geoms = []
        for g in range(model.ngeom):
            if model.body(model.geom_bodyid[g]).name in body_names and \
                    model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                # float64 copy: MuJoCo stores vertices as float32.
                self.geoms.append((g, model.mesh_vert[start:start + count].astype(np.float64)))

    def __call__(self, data):
        return min(
            (verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min()
            for g, verts in self.geoms
        )
