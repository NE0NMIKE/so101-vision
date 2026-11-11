"""Check that each motor ID sits in the joint it is named after.

A bus scan only proves IDs 1-6 exist. It can't tell if, say, the elbow and wrist
motors were swapped during assembly. This script reads raw encoder positions
with torque OFF while you move the joints by hand, one at a time, from the base
to the gripper. It reports the order in which motors first moved; if the arm is
wired correctly that order is 1, 2, 3, 4, 5, 6.

    python scripts/check_joint_mapping.py --port /dev/cu.usbmodemXXXX

No calibration is needed: it uses raw encoder ticks (0-4095 per turn).
"""

import argparse
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

TICKS_PER_TURN = 4096
MOVE_THRESHOLD_TICKS = 150  # About 13 degrees: larger than any knock-on wobble.
SAMPLE_PERIOD_S = 0.02      # 50 Hz.


def wrapped_delta(now, start):
    """Signed change in ticks, treating 4095 -> 0 as a step of +1, not -4095."""
    return (now - start + TICKS_PER_TURN // 2) % TICKS_PER_TURN - TICKS_PER_TURN // 2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True)
    parser.add_argument("--timeout", type=float, default=150, help="Seconds to wait for all joints")
    args = parser.parse_args()

    bus = SOFollower(SOFollowerRobotConfig(port=args.port)).bus
    bus.connect()          # Handshake: fails if any of IDs 1-6 is missing.
    bus.disable_torque()   # Let the joints move freely by hand.

    try:
        start = bus.sync_read("Present_Position", normalize=False, num_retry=5)
        print("Start (raw ticks):", start, flush=True)
        print("Move each joint in turn, base to gripper...", flush=True)

        t0 = time.monotonic()
        first_moved = {}  # motor -> seconds after start
        while len(first_moved) < len(bus.motors) and time.monotonic() - t0 < args.timeout:
            now = bus.sync_read("Present_Position", normalize=False, num_retry=5)
            for motor, pos in now.items():
                if motor not in first_moved and abs(wrapped_delta(pos, start[motor])) > MOVE_THRESHOLD_TICKS:
                    first_moved[motor] = time.monotonic() - t0
                    print(f"  {time.monotonic() - t0:6.1f}s  moved: {motor} (ID {bus.motors[motor].id})",
                          flush=True)
            time.sleep(SAMPLE_PERIOD_S)

        order = sorted(first_moved, key=first_moved.get)
        ids = [bus.motors[m].id for m in order]
        print("\nOrder of first movement (IDs):", ids)
        if missing := [m for m in bus.motors if m not in first_moved]:
            print("Never moved enough to detect:", missing)
        print("PASS: wiring matches joint names." if ids == [1, 2, 3, 4, 5, 6]
              else "CHECK: order differs from 1-6. Either a joint was moved out of order, or motors are swapped.")
    finally:
        bus.disconnect()  # Torque off on exit.


if __name__ == "__main__":
    main()
