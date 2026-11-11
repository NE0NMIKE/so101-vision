"""Move a simulated SO-101 gripper to XYZ targets, given in metres.

Run without a window:
    python so101_sim_move.py --scene assets/urdf/scene.xml --target 0.20 0.00 0.15

Run with a 3D window (on macOS, use mjpython instead of python):
    mjpython so101_sim_move.py --scene assets/urdf/scene.xml --viewer \
        --target 0.20 0.00 0.15 --target 0.25 0.10 0.10

The script has two separate parts:
1. solve_ik() is pure maths. It finds joint angles that put the gripper on the target.
2. move_to() is physics. It drives the motors to those angles and measures the result.
Keeping them separate matters because the real arm will reuse the same idea:
the maths can say "perfect" while the physical arm still sags or overshoots.
"""

import argparse
import time

import mujoco
import numpy as np

# Joint names exactly as they appear in the SO-101 MuJoCo model.
# The gripper is left out because opening it barely moves the point we control.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# The point on the arm we want to position, defined in the model.
TIP_SITE = "gripperframe"

# A bent, comfortable starting pose in radians. Starting with the arm folded
# avoids the straight-arm singularity, where some directions become unreachable.
START_POSE = np.array([0.0, -0.8, 0.8, 0.5, 0.0])

IK_TOLERANCE_M = 0.001  # IK counts as solved within 1 mm.
IK_MAX_ITERATIONS = 500
IK_DAMPING = 0.02       # Higher values are more stable near singularities but slower.
IK_MAX_STEP_RAD = 0.05  # Largest joint change allowed in one IK iteration.


def joint_indices(model):
    """Return where each arm joint sits in the qpos and qvel arrays."""
    joint_ids = [model.joint(name).id for name in ARM_JOINTS]
    qpos_ids = model.jnt_qposadr[joint_ids]  # Position array indices.
    qvel_ids = model.jnt_dofadr[joint_ids]   # Velocity and Jacobian column indices.
    return joint_ids, qpos_ids, qvel_ids


def solve_ik(model, target_xyz, start_q):
    """Find arm joint angles that place the tip at target_xyz.

    Uses damped least squares:   dq = J^T (J J^T + lambda^2 I)^-1 e
      e  = position error, tip to target (3 values)
      J  = Jacobian: how far the tip moves per radian of each joint (3 x 5)
      dq = small joint change that reduces the error
    The damping term lambda keeps dq sensible when J is close to singular.

    Returns (joint_angles, final_error_in_metres, solved_flag).
    """
    # A scratch copy of the simulation state, so IK can't disturb the physics.
    scratch = mujoco.MjData(model)
    joint_ids, qpos_ids, qvel_ids = joint_indices(model)
    site_id = model.site(TIP_SITE).id
    lower, upper = model.jnt_range[joint_ids].T

    q = np.clip(start_q, lower, upper)
    jacobian = np.zeros((3, model.nv))
    error_norm = np.inf

    for _ in range(IK_MAX_ITERATIONS):
        # Update where everything is for the current guess. No physics here.
        scratch.qpos[qpos_ids] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)

        error = target_xyz - scratch.site_xpos[site_id]
        error_norm = np.linalg.norm(error)
        if error_norm < IK_TOLERANCE_M:
            return q, error_norm, True

        # Fill in the translational Jacobian. None means rotation isn't requested.
        mujoco.mj_jacSite(model, scratch, jacobian, None, site_id)
        J = jacobian[:, qvel_ids]

        # Solve the 3 x 3 system instead of inverting a matrix directly.
        damped = J @ J.T + IK_DAMPING**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(damped, error)

        # Limit the step, then keep every joint inside its mechanical range.
        dq = np.clip(dq, -IK_MAX_STEP_RAD, IK_MAX_STEP_RAD)
        q = np.clip(q + dq, lower, upper)

    return q, error_norm, False


def min_jerk(s):
    """Map progress s from 0 to 1 onto a curve that starts and stops smoothly."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def move_to(model, data, q_goal, duration_s, viewer=None):
    """Drive the position actuators from the current pose to q_goal."""
    _, qpos_ids, _ = joint_indices(model)
    actuator_ids = [model.actuator(name).id for name in ARM_JOINTS]
    q_start = data.qpos[qpos_ids].copy()

    # Physics runs in many small steps of model.opt.timestep seconds.
    n_steps = int(duration_s / model.opt.timestep)
    settle_steps = int(1.0 / model.opt.timestep)  # Extra 1 s to come to rest.

    for step in range(n_steps + settle_steps):
        progress = min(step / n_steps, 1.0)
        data.ctrl[actuator_ids] = q_start + min_jerk(progress) * (q_goal - q_start)
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)  # Run at roughly real-time speed.


def go_to_target(model, data, target_xyz, duration_s, viewer=None):
    """Solve IK, execute the move if it's reachable, and report the result."""
    _, qpos_ids, _ = joint_indices(model)
    site_id = model.site(TIP_SITE).id
    print(f"\nTarget (m): {np.round(target_xyz, 3)}")

    q_goal, ik_error, solved = solve_ik(model, target_xyz, data.qpos[qpos_ids].copy())
    if not solved:
        # Refuse rather than move to a "best effort" pose nobody asked for.
        print(f"  Not reachable: IK stopped {ik_error * 1000:.1f} mm away. Arm not moved.")
        return False

    print(f"  Joint goal (deg): {np.round(np.degrees(q_goal), 1)}")
    move_to(model, data, q_goal, duration_s, viewer)
    error_mm = np.linalg.norm(target_xyz - data.site_xpos[site_id]) * 1000
    print(f"  IK error: {ik_error * 1000:.2f} mm | Error after physics: {error_mm:.2f} mm")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", required=True, help="Path to the SO-101 scene.xml")
    parser.add_argument("--target", nargs=3, type=float, action="append", required=True,
                        metavar=("X", "Y", "Z"), help="Target in metres; repeat for a sequence")
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds per move")
    parser.add_argument("--viewer", action="store_true", help="Open the 3D window")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    # Place the arm in the start pose and tell the actuators to hold it.
    _, qpos_ids, _ = joint_indices(model)
    actuator_ids = [model.actuator(name).id for name in ARM_JOINTS]
    data.qpos[qpos_ids] = START_POSE
    data.ctrl[actuator_ids] = START_POSE
    mujoco.mj_forward(model, data)

    if args.viewer:
        import mujoco.viewer as mj_viewer  # Imported here so headless runs need no display.
        with mj_viewer.launch_passive(model, data) as viewer:
            for target in args.target:
                go_to_target(model, data, np.array(target), args.duration, viewer)
            while viewer.is_running():  # Keep the window open until it's closed.
                time.sleep(0.05)
    else:
        for target in args.target:
            go_to_target(model, data, np.array(target), args.duration)


if __name__ == "__main__":
    main()
