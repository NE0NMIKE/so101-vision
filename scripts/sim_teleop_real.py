"""Drive the real SO-101 from MuJoCo: move the sim arm, the real arm follows.

Open the viewer's right-hand "Control" panel and drag the actuator sliders. The
sim arm moves under physics, and its joint angles are streamed to the real arm
at 30 Hz through these safety layers:
  1. No jump at start: the sim starts at the real pose, and each motor's goal is
     set to its present position (verified by read-back) before torque comes on.
  2. Speed limit (--max-speed): the COMMAND may move at most this fast, however
     fast the sliders move. It limits the goal, not how far the arm may lag.
  3. Error cap (--max-error): LeRobot's max_relative_target keeps each command
     within this many degrees of the MEASURED position. With the motors' P
     controller (effort ~ gain x error) this also caps force if the arm is blocked,
     so it must be large enough to lift the outstretched arm against gravity.
  4. Virtual floor (--floor-mm): commands that bring the gripper or wrist this
     close to the table are blocked unless they move upward.
  5. Safe exit: on close, the arm glides back to its start pose, then torque off.

Dry run (default): reads the real arm, sends nothing, prints what it would send.
    mjpython scripts/sim_teleop_real.py --config config/so101_follower.yaml
Live: the real arm moves.
    mjpython scripts/sim_teleop_real.py --config config/so101_follower.yaml --live
"""

import argparse
import logging
import time

import mujoco
import numpy as np

from lerobot.robots.so_follower import SOFollower
from real_sim_alignment import (
    ARM_JOINTS, LowestPoint, load_offsets, load_robot_config, real_to_sim, set_pose,
    sim_gripper_to_real, sim_to_real,
)

CONTROL_HZ = 30
GUARDED_BODIES = {"wrist", "gripper", "moving_jaw_so101_v1"}
RETURN_HOME_S = 3.0
STALL_DEG = 5.0      # A joint this far behind its command...
STALL_S = 1.0        # ...for this long is reported as stalled.


def min_jerk(s):
    """Smooth 0 -> 1 progress curve: zero velocity and acceleration at both ends."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Sim:
    def __init__(self, scene):
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.qpos_index = {j: self.model.jnt_qposadr[self.model.joint(j).id] for j in [*ARM_JOINTS, "gripper"]}
        self.actuator_index = {j: self.model.actuator(j).id for j in [*ARM_JOINTS, "gripper"]}
        self.guarded_low = LowestPoint(self.model, GUARDED_BODIES)
        self.scratch = mujoco.MjData(self.model)
        set_pose(self.model, self.scratch, {})
        self.table_z = LowestPoint(self.model, {"base"})(self.scratch)
        self.steps_per_tick = max(1, round(1 / (CONTROL_HZ * self.model.opt.timestep)))

    def start_at(self, q):
        """Place the sim arm at q and make the actuators hold it there."""
        for joint, angle in q.items():
            self.data.qpos[self.qpos_index[joint]] = angle
            self.data.ctrl[self.actuator_index[joint]] = angle
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        for _ in range(self.steps_per_tick):  # One control tick of physics.
            mujoco.mj_step(self.model, self.data)

    def joint_angles(self):
        return {j: float(self.data.qpos[i]) for j, i in self.qpos_index.items()}

    def height_above_table_mm(self, q):
        """Lowest guarded point for joint angles q, on a scratch copy so physics is untouched."""
        set_pose(self.model, self.scratch, q)
        return (self.guarded_low(self.scratch) - self.table_z) * 1000


def command_from_sim(sim, q, offsets):
    action = {f"{j}.pos": v for j, v in sim_to_real(q, offsets).items()}
    action["gripper.pos"] = sim_gripper_to_real(sim.model, q["gripper"])
    return action


def rate_limit(target, previous, max_step):
    """Move each value from previous toward target by at most max_step."""
    return {k: previous[k] + float(np.clip(target[k] - previous[k], -max_step, max_step)) for k in target}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", default="assets/urdf/scene.xml")
    parser.add_argument("--live", action="store_true", help="Enable torque and move the real arm")
    parser.add_argument("--max-speed", type=float, default=120.0,
                        help="Command speed limit, deg/s (gripper: %%/s)")
    parser.add_argument("--max-error", type=float, default=15.0,
                        help="Max command-vs-measured difference per joint, deg")
    parser.add_argument("--floor-mm", type=float, default=15.0, help="Virtual floor height above the table")
    parser.add_argument("--headless-seconds", type=float, default=0,
                        help="Dry-run test without a window for N seconds (sliders unavailable)")
    args = parser.parse_args()
    if args.live and args.headless_seconds:
        parser.error("--live needs the viewer window (so you can close it to stop).")

    # LeRobot warns on every clipped command; we report clipping and stalls ourselves.
    logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)

    config = load_robot_config(args.config)
    config.max_relative_target = args.max_error  # Layer 3, enforced inside LeRobot's send_action.
    robot = SOFollower(config)
    offsets = load_offsets(robot.calibration_fpath)
    sim = Sim(args.scene)
    max_step = args.max_speed / CONTROL_HZ
    print(f"Limits: speed {args.max_speed:.0f} deg/s ({max_step:.1f} deg/tick), "
          f"error cap {args.max_error:.0f} deg, virtual floor {args.floor_mm:.0f} mm")

    bus = robot.bus
    bus.connect()
    bus.disable_torque()
    start_reading = bus.sync_read("Present_Position", num_retry=3)  # Degrees / gripper 0-100.
    sim.start_at(real_to_sim(sim.model, start_reading, offsets))
    home_action = {f"{m}.pos": v for m, v in start_reading.items()}
    print("Start pose (real, deg):", {m: round(v, 1) for m, v in start_reading.items()})

    if args.live:
        # Layer 1: goal := present (raw ticks) so enabling torque doesn't jump to a stale goal.
        present_raw = bus.sync_read("Present_Position", normalize=False, num_retry=3)
        bus.sync_write("Goal_Position", present_raw, normalize=False)
        # Sync writes are not acknowledged, so read the goals back before trusting them.
        goal_raw = bus.sync_read("Goal_Position", normalize=False, num_retry=3)
        if bad := [m for m in present_raw if abs(goal_raw[m] - present_raw[m]) > 5]:
            bus.disconnect()
            raise SystemExit(f"Goal_Position didn't match present position for {bad}. Not enabling torque.")
        robot.configure()  # Position mode + LeRobot PID gains; re-enables torque at the end.
        print("TORQUE ON. The arm is holding its pose. Close the viewer window to stop.")
    else:
        print("DRY RUN: nothing will be sent. Use --live to move the real arm.")

    state = {
        "commanded": dict(home_action),   # Last rate-limited command (what the arm is told).
        "last_safe_height": sim.height_above_table_mm(sim.joint_angles()),
        "floor_blocked": False,
        "clipped": False,
        "stall_since": {},
    }

    def tick():
        sim.step()
        q = sim.joint_angles()
        height = sim.height_above_table_mm(q)
        # Layer 4: below the virtual floor, only allow moves that go up.
        if height < args.floor_mm and height < state["last_safe_height"] - 0.5:
            if not state["floor_blocked"]:
                print(f"\nVirtual floor: gripper would be {height:.0f} mm above the table. Holding.")
            state["floor_blocked"] = True
            target = state["commanded"]
        else:
            state["floor_blocked"] = False
            state["last_safe_height"] = height
            target = command_from_sim(sim, q, offsets)
        # Layer 2: the command glides toward the target at no more than max_speed.
        command = rate_limit(target, state["commanded"], max_step)
        state["commanded"] = command
        if args.live:
            sent = robot.send_action(command)  # Layer 3 may clip it against the measured position.
            state["clipped"] = any(abs(sent[k] - command[k]) > 1e-6 for k in command)

    def status():
        command = state["commanded"]
        present = bus.sync_read("Present_Position", num_retry=3)
        now = time.monotonic()
        stalled = []
        for m in ARM_JOINTS:
            if args.live and abs(command[f"{m}.pos"] - present[m]) > STALL_DEG:
                since = state["stall_since"].setdefault(m, now)
                if now - since > STALL_S:
                    stalled.append(m)
            else:
                state["stall_since"].pop(m, None)
        lag = max(abs(command[f"{m}.pos"] - present[m]) for m in ARM_JOINTS)
        cmd = " ".join(f"{m.split('_')[0][:5]}{command[f'{m}.pos']:+6.1f}" for m in ARM_JOINTS)
        flags = (" CLIPPED" if state["clipped"] else "") + (f" STALL: {','.join(stalled)}" if stalled else "")
        return f"\rcmd {cmd} grip{command['gripper.pos']:5.1f}% | lag {lag:5.1f} deg{flags:<30}"

    try:
        if args.headless_seconds:
            end = time.monotonic() + args.headless_seconds
            while time.monotonic() < end:
                tick()
                time.sleep(1 / CONTROL_HZ)
            print(status().strip())
            return

        import mujoco.viewer as mj_viewer

        with mj_viewer.launch_passive(sim.model, sim.data) as viewer:
            last_print = 0.0
            while viewer.is_running():
                t0 = time.monotonic()
                with viewer.lock():  # The UI writes ctrl from the sliders; don't race it.
                    tick()
                viewer.sync()
                if time.monotonic() - last_print > 0.25:
                    print(status(), end="", flush=True)
                    last_print = time.monotonic()
                time.sleep(max(0.0, 1 / CONTROL_HZ - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if args.live:
            # Layer 5: glide back to the start pose, then release.
            print("\nReturning to start pose...")
            begin = dict(state["commanded"])
            n = int(RETURN_HOME_S * CONTROL_HZ)
            for i in range(1, n + 1):
                s = min_jerk(i / n)
                robot.send_action({k: begin[k] + s * (home_action[k] - begin[k]) for k in home_action})
                time.sleep(1 / CONTROL_HZ)
            # The error cap can leave the arm behind the glide: keep commanding home until it arrives.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                robot.send_action(home_action)
                present = bus.sync_read("Present_Position", num_retry=3)
                if max(abs(present[m] - start_reading[m]) for m in ARM_JOINTS) < 2.0:
                    break
                time.sleep(1 / CONTROL_HZ)
        bus.disconnect()  # Torque off.
        print("\nDisconnected, torque off.")


if __name__ == "__main__":
    main()
