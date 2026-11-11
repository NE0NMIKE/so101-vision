"""Digital twin: the MuJoCo arm copies the real arm as you move it by hand.

This checks that the real arm and the simulation agree on the two things a
calibration file can't prove:
  * direction: does +30 deg on the motor mean +30 deg in the model?
  * zero:      does 0 deg on the motor look like 0 deg in the model?
Directions are set in SIGN and zero corrections in config/sim_joint_offsets.yaml
(both in real_sim_alignment.py).

Torque stays OFF the whole time, so the arm is free to move (and will fall
under gravity, so support it).

On macOS the 3D window needs mjpython:
    mjpython scripts/mirror_real_to_sim.py --config config/so101_follower.yaml
One reading, no window:
    python scripts/mirror_real_to_sim.py --config config/so101_follower.yaml --once
Compare with the uncorrected model:
    ... --no-offsets
"""

import argparse
import time

import mujoco

from lerobot.robots.so_follower import SOFollower
from real_sim_alignment import ARM_JOINTS, load_offsets, load_robot_config, real_to_sim, set_pose

UPDATE_HZ = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="YAML with the robot section")
    parser.add_argument("--scene", default="assets/urdf/scene.xml")
    parser.add_argument("--once", action="store_true", help="Print one reading and exit, no window")
    parser.add_argument("--no-offsets", action="store_true", help="Ignore fitted zero offsets")
    args = parser.parse_args()

    robot = SOFollower(load_robot_config(args.config))
    if not robot.calibration:
        raise SystemExit(f"No calibration file at {robot.calibration_fpath}. Run lerobot-calibrate first.")
    offsets = {j: 0.0 for j in ARM_JOINTS} if args.no_offsets else load_offsets(robot.calibration_fpath)
    print("Zero offsets (deg):", {j: v for j, v in offsets.items() if v})

    # Use the bus directly instead of robot.connect(): connect() also runs configure(),
    # which can switch torque back on. Here we only want to read.
    bus = robot.bus
    bus.connect()
    bus.disable_torque()
    print("Calibration in motors matches file:", bus.is_calibrated)

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    def update():
        reading = bus.sync_read("Present_Position", num_retry=3)
        set_pose(model, data, real_to_sim(model, reading, offsets))  # Pose only, no physics.
        return reading

    def show(reading):
        cells = [f"{j[:11]:>11} {reading[j]:+7.1f}" for j in ARM_JOINTS]
        return " |".join(cells) + f" | gripper {reading['gripper']:5.1f}%"

    try:
        if args.once:
            print(show(update()))
            return

        import mujoco.viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            last_print = 0.0
            while viewer.is_running():
                reading = update()
                viewer.sync()
                if time.monotonic() - last_print > 0.2:
                    print("\r" + show(reading), end="", flush=True)
                    last_print = time.monotonic()
                time.sleep(1 / UPDATE_HZ)
    finally:
        bus.disconnect()  # Torque off on the way out.
        print()


if __name__ == "__main__":
    main()
