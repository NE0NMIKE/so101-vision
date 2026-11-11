"""Give each SO-101 follower motor its ID, one motor at a time.

Every new Feetech STS3215 leaves the factory as ID 1. If two of them share the
bus they both answer to ID 1 and their replies collide, so each motor must be
alone on the controller board while it gets its ID.

Scan the bus (read-only, writes nothing):
    python scripts/setup_motor_ids.py --port /dev/cu.usbmodemXXXX --scan

Set one motor's ID (only that motor plugged into the board):
    python scripts/setup_motor_ids.py --port /dev/cu.usbmodemXXXX --motor gripper

This does the same job as `lerobot-setup-motors`, but one motor per run, so
each step can be checked before moving to the next.
"""

import argparse

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig


def make_bus(port):
    """Build the motor bus exactly as LeRobot's SO-101 follower does."""
    robot = SOFollower(SOFollowerRobotConfig(port=port))
    return robot.bus


def scan(bus):
    """Ping every ID at every baud rate the STS3215 supports and report who answers."""
    definition = bus._definition("gripper")  # All six motors share the STS3215 definition.
    bus.connect(handshake=False)  # Open the port without checking for motors 1-6.
    try:
        found_any = False
        for baudrate in sorted(definition.baudrates, reverse=True):
            bus.set_baudrate(baudrate)
            answers = bus._scan(definition)  # {id: model_number}
            if answers:
                found_any = True
                print(f"{baudrate:>9} baud: IDs {sorted(answers)}  (model numbers {set(answers.values())})")
        if not found_any:
            print("No motor answered. Check the power supply and the cable to the motor.")
    finally:
        bus.set_baudrate(bus.default_baudrate)
        bus.disconnect(disable_torque=False)  # Nothing is guaranteed to be at IDs 1-6 yet.


def set_id(bus, motor):
    """Find the single connected motor, then write its new ID and baud rate."""
    target_id = bus.motors[motor].id
    print(f"Looking for one motor on the bus to become '{motor}' (ID {target_id})...")
    bus.setup_motor(motor)  # Finds it, unlocks EEPROM, writes ID + 1 Mbaud, verifies with a ping.
    print(f"Done: '{motor}' now answers as ID {target_id} at {bus.default_baudrate} baud.")
    print(f"Label this motor '{target_id} {motor}' before unplugging it.")
    bus.disconnect(disable_torque=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True, help="Serial port of the controller board")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true", help="List motors on the bus, change nothing")
    group.add_argument("--motor", help="Motor name to assign, e.g. gripper or shoulder_pan")
    args = parser.parse_args()

    bus = make_bus(args.port)
    if args.scan:
        scan(bus)
    else:
        if args.motor not in bus.motors:
            parser.error(f"--motor must be one of {list(bus.motors)}")
        set_id(bus, args.motor)


if __name__ == "__main__":
    main()
