"""Fit per-joint zero offsets so the MuJoCo arm matches the real one.

Idea: the table is a flat plane at a known height (the bottom of the base).
Every pose in which the gripper just touches the table gives one equation:
    lowest point of the sim gripper, with corrected joint angles  =  table height
With ~10 varied poses we solve for constant offsets on the joints that change
the gripper's height (shoulder_lift, elbow_flex, wrist_flex) by least squares.

Step 1, record touch poses (interactive, torque OFF):
    python scripts/table_touch_calibration.py record --config config/so101_follower.yaml
Step 2, fit, report and (with --write) save config/sim_joint_offsets.yaml:
    python scripts/table_touch_calibration.py fit --config config/so101_follower.yaml --write
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from lerobot.robots.so_follower import SOFollower
from real_sim_alignment import (
    ARM_JOINTS, LowestPoint, load_offsets, load_robot_config, real_to_sim, save_offsets, set_pose,
)

FIT_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]  # The joints that move the gripper up/down.
CONTACT_BODIES = {"gripper"}   # Fixed jaw + gripper housing. The moving jaw stays on top, off the table.
TABLE_BODIES = {"base"}        # The base sits on the same table, so its bottom defines table height.
SAMPLES_PER_CAPTURE = 10       # Averaged to reduce encoder jitter.
MAX_ROLL_DEG = 30              # Beyond this the moving jaw may be the part touching the table.


def zero_offsets():
    return {joint: 0.0 for joint in ARM_JOINTS}


class TouchModel:
    """Predicts, for a recorded pose, how far the sim gripper is above (+) or below (-) the table."""

    def __init__(self, scene):
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.gripper_low = LowestPoint(self.model, CONTACT_BODIES)
        set_pose(self.model, self.data, {})  # All joints at 0: find the base bottom.
        self.table_z = LowestPoint(self.model, TABLE_BODIES)(self.data)

    def gap_mm(self, reading, offsets):
        set_pose(self.model, self.data, real_to_sim(self.model, reading, offsets))
        return (self.gripper_low(self.data) - self.table_z) * 1000


# ---------------------------------------------------------------- record

def record(args):
    robot = SOFollower(load_robot_config(args.config))
    touch = TouchModel(args.scene)
    out = Path(args.data)
    samples = json.loads(out.read_text()) if out.exists() else []
    print(f"{len(samples)} poses already in {out}. New ones are appended.\n")

    bus = robot.bus
    bus.connect()
    bus.disable_torque()
    print("Torque is OFF. For each pose: hold the arm so the fixed claw JUST touches the table,")
    print("bearing the arm's weight with your hand (don't let it lean on the table).")
    print("Keep the movable jaw on top. Vary reach, elbow bend and wrist angle between poses.\n")
    try:
        while True:
            cmd = input(f"Pose {len(samples) + 1}: ENTER = capture, u = undo last, q = finish > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "u":
                if samples:
                    samples.pop()
                    out.write_text(json.dumps(samples, indent=1))
                    print(f"  removed. {len(samples)} poses left.")
                continue

            reads = []
            for _ in range(SAMPLES_PER_CAPTURE):
                reads.append(bus.sync_read("Present_Position", num_retry=3))
                time.sleep(0.02)
            reading = {m: float(np.mean([r[m] for r in reads])) for m in reads[0]}
            jitter = max(np.ptp([r[m] for r in reads]) for m in ARM_JOINTS)

            if abs(reading["wrist_roll"]) > MAX_ROLL_DEG:
                print(f"  SKIPPED: wrist_roll is {reading['wrist_roll']:+.0f} deg. Turn the jaw back on top.")
                continue
            if jitter > 1.0:
                print(f"  SKIPPED: arm moved during capture ({jitter:.1f} deg). Hold it still and retry.")
                continue

            samples.append(reading)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(samples, indent=1))  # Save every time, so nothing is lost.
            angles = "  ".join(f"{j.split('_')[0]}{reading[j]:+6.1f}" for j in ARM_JOINTS[:4])
            print(f"  saved: {angles}  | sim gap now {touch.gap_mm(reading, zero_offsets()):+6.1f} mm")
    finally:
        bus.disconnect()
    print(f"\n{len(samples)} poses saved to {out}.")


# ---------------------------------------------------------------- fit

def fit(args):
    robot = SOFollower(load_robot_config(args.config))
    samples = json.loads(Path(args.data).read_text())
    if args.exclude:
        print(f"Excluding poses {args.exclude} (1-based) from the fit.")
        samples = [s for i, s in enumerate(samples, 1) if i not in args.exclude]
    touch = TouchModel(args.scene)
    n, p = len(samples), len(FIT_JOINTS)
    if n < 2 * p:
        raise SystemExit(f"Only {n} poses. Record at least {2 * p} (10 is better) before fitting.")

    def offsets_from(x):
        offsets = zero_offsets()
        offsets.update(dict(zip(FIT_JOINTS, x)))
        return offsets

    def residuals(x):
        """One number per pose: sim gap to the table in mm. A perfect fit makes all of them 0."""
        offsets = offsets_from(x)
        return np.array([touch.gap_mm(s, offsets) for s in samples])

    # least_squares minimises sum(residuals^2) by adjusting x, the offsets in degrees.
    # Under the hood (method="trf") it linearises residuals with a Jacobian and takes
    # damped steps, the same damped-least-squares idea as the IK in so101_sim_move.py.
    # diff_step: the Jacobian is estimated by nudging each offset by ~0.001 deg.
    result = least_squares(residuals, x0=np.zeros(p), diff_step=1e-3, x_scale=1.0)

    before, after = residuals(np.zeros(p)), result.fun
    rms = lambda r: float(np.sqrt(np.mean(r**2)))

    # Uncertainty: covariance ~ s^2 (J^T J)^-1, with s^2 the leftover residual variance.
    J = result.jac
    dof = max(n - p, 1)
    s2 = float(after @ after) / dof
    cov = s2 * np.linalg.pinv(J.T @ J)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    print(f"Table height in sim (bottom of base): {touch.table_z * 1000:+.1f} mm")
    print(f"Poses: {n}   Unknowns: {p}\n")
    print(f"{'pose':>4} {'before (mm)':>12} {'after (mm)':>11}")
    for i, (b, a) in enumerate(zip(before, after), 1):
        print(f"{i:>4} {b:>+12.1f} {a:>+11.1f}")
    print(f"\nRMS gap: {rms(before):.1f} mm before -> {rms(after):.1f} mm after\n")
    print("Fitted offsets (sim_deg = real_deg + offset):")
    for joint, value, sd in zip(FIT_JOINTS, result.x, std):
        print(f"  {joint:<14} {value:+6.2f} deg  (+/- {sd:.2f})")
    print("\nCorrelation between offsets (|r| near 1 = poses don't separate these joints well):")
    print("               " + "".join(f"{j[:11]:>12}" for j in FIT_JOINTS))
    for j, row in zip(FIT_JOINTS, corr):
        print(f"  {j:<13}" + "".join(f"{r:>12.2f}" for r in row))

    plot(before, after, Path(args.data).with_suffix(".png"))

    if args.write:
        notes = f"{n} table-touch poses, RMS gap {rms(before):.1f} -> {rms(after):.1f} mm"
        save_offsets(offsets_from(result.x), robot.calibration_fpath, notes)
        print("\nSaved. Check it loads:", load_offsets(robot.calibration_fpath))
    else:
        print("\nNot saved. Re-run with --write to save config/sim_joint_offsets.yaml.")


def plot(before, after, path):
    """Grouped bars: each pose's gap to the table before and after the fit."""
    import matplotlib
    matplotlib.use("Agg")  # Draw to a file, no window.
    import matplotlib.pyplot as plt

    surface, ink, muted = "#fcfcfb", "#1f1e1d", "#73726c"
    x = np.arange(1, len(before) + 1)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.bar(x - width / 2 - 0.01, before, width, color="#2a78d6", label="Before fit")
    ax.bar(x + width / 2 + 0.01, after, width, color="#eb6834", label="After fit")
    ax.axhline(0, color=ink, linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("Touch pose", color=muted)
    ax.set_ylabel("Sim gripper to table (mm)", color=muted)
    ax.set_title("Table-touch calibration: gap per pose (0 = touching)", color=ink, loc="left")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nPlot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["record", "fit"])
    parser.add_argument("--config", required=True, help="YAML with the robot section")
    parser.add_argument("--scene", default="assets/urdf/scene.xml")
    parser.add_argument("--data", default="data/table_touch.json", help="Recorded poses")
    parser.add_argument("--write", action="store_true", help="fit: save the offsets")
    parser.add_argument("--exclude", type=int, nargs="*", default=[],
                        help="fit: pose numbers (1-based) to leave out, e.g. --exclude 7")
    args = parser.parse_args()
    record(args) if args.command == "record" else fit(args)


if __name__ == "__main__":
    main()
