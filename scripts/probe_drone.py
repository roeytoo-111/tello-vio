#!/usr/bin/env python3
"""Ask the drone what it is and what ground-truth options it supports.

    python3 scripts/probe_drone.py

Run this with the Tello powered on and this machine joined to its WiFi AP.
It talks raw UDP -- no ROS, no djitellopy -- so it works before anything is
built and cannot be confused by the rest of the stack.

The question it answers: mission-pad ground truth needs a Tello EDU or
RoboMaster TT on SDK 2.0+. A standard Tello on SDK 1.3 has no pad detection
hardware, and its only ground-truth option is an external reference.
"""
import socket
import sys

ADDR = ("192.168.10.1", 8889)
TIMEOUT = 4.0


def is_ok(reply) -> bool:
    """True only for an accepted command.

    The Tello signals rejection in several ways and ``error`` is only one of
    them: a standard Tello answers an SDK 2.0 command with
    ``"unknown command: mon"``. Testing for an ``error`` prefix means every
    *other* form of rejection reads as SUCCESS -- which is exactly how this
    probe once told a standard Tello that it supported Mission Pads.

    So: accept exactly the documented success token and treat everything else,
    including silence, as refusal. Note this is stricter than djitellopy's own
    ``'ok' in response.lower()`` substring test, which would also accept a
    reply that merely contains those two letters.
    """
    return reply is not None and str(reply).strip().lower() == "ok"


def main() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("", 8889))
    except OSError as e:
        print(f"ERROR: cannot bind UDP 8889 ({e}).")
        print("  Something else is using it -- most likely a tello driver still")
        print("  running. Stop it first:  pkill -9 -f 'lib/tello/tello'")
        return 2
    s.settimeout(TIMEOUT)

    def ask(cmd):
        try:
            s.sendto(cmd.encode(), ADDR)
            data, _ = s.recvfrom(1024)
            return data.decode(errors="replace").strip()
        except socket.timeout:
            return None
        except OSError as e:
            return f"<socket error: {e}>"

    print(f"probing {ADDR[0]} ...\n")
    if not is_ok(ask("command")):
        print("No response to 'command'.")
        print("  The drone is off, asleep, or this machine is not on its WiFi AP.")
        print("  On WSL2 check that WINDOWS is joined to TELLO-XXXXXX.")
        s.close()
        return 1

    sdk = ask("sdk?")
    serial = ask("sn?")
    mon = ask("mon")
    supported = is_ok(mon)
    if supported:
        ask("moff")          # leave the drone as we found it

    sdk_txt = sdk if is_ok(sdk) or (sdk and "unknown" not in sdk.lower()) \
        else f"{sdk or 'no reply'}  (=> SDK 1.3, standard Tello)"

    print(f"  SDK version   : {sdk_txt}")
    print(f"  serial number : {serial if serial else 'not supported'}")
    print(f"  'mon' reply   : {mon if mon is not None else 'no response'}")
    print()
    if supported:
        print("MISSION PADS: SUPPORTED.")
        print("  Best ground truth available to you. Print a Mission Pad, then:")
        print("    ros2 launch tello_vio vio.launch.py mission_pad:=true")
        print("    ros2 bag record -o flight1 /tello_vio/odom /mission_pad_pose")
        print("    ros2 run tello_vio evaluate_bag --ros-args -p bag:=flight1 \\")
        print("        -p gt_topic:=/mission_pad_pose -p plot:=err.png")
        print("  Verify the axis signs once -- see the README.")
    else:
        print("MISSION PADS: NOT SUPPORTED by this airframe.")
        print("  Pad detection needs a Tello EDU or RoboMaster TT on SDK 2.0+.")
        print("  Use the printed ArUco marker instead:")
        print("    ros2 run tello_vio make_marker --ros-args -p output:=marker.png")
        print("    ros2 run tello_vio ground_truth --ros-args -p marker_size_m:=0.195")
    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
