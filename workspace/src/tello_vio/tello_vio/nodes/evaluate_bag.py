#!/usr/bin/env python3
"""Score a recorded flight against fiducial ground truth and plot the error.

    # 1. record a flight (VIO stack + ground_truth running)
    ros2 bag record -o flight1 /tello_vio/odom /tello_ground_truth/pose

    # 2. score it
    ros2 run tello_vio evaluate_bag --ros-args -p bag:=flight1 -p plot:=err.png

Prints ATE (SE3 and Sim3), RPE and drift, and writes a three-panel figure:
trajectory top-down, per-axis position, and position error over time.
"""

from __future__ import annotations

import numpy as np


def read_bag(path: str, est_topic: str, gt_topic: str):
    """Read two pose-ish topics out of a rosbag2 into timestamped arrays."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for want in (est_topic, gt_topic):
        if want not in types:
            raise SystemExit(
                f"topic {want!r} not in bag. Recorded topics:\n  " +
                "\n  ".join(sorted(types)))

    out = {est_topic: ([], []), gt_topic: ([], [])}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in out:
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        h = msg.header
        pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
        out[topic][0].append(h.stamp.sec + h.stamp.nanosec * 1e-9)
        out[topic][1].append([pose.position.x, pose.position.y, pose.position.z])
    return ({k: (np.asarray(v[0]), np.asarray(v[1]).reshape(-1, 3))
             for k, v in out.items()})


def plot(t, est, gt, err, out_png: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(title, fontsize=12, weight="bold")

    ax[0].plot(gt[:, 0], gt[:, 1], "k-", lw=2, label="ground truth")
    ax[0].plot(est[:, 0], est[:, 1], color="#1f6feb", lw=1.6, label="VIO (aligned)")
    ax[0].scatter(*gt[0, :2], c="green", s=60, zorder=5, label="start")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].set_title("trajectory, top-down"); ax[0].axis("equal")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    t0 = t - t[0]
    for k, (lab, c) in enumerate(zip("xyz", ["#1f6feb", "#c2600f", "#1f8a4c"])):
        ax[1].plot(t0, gt[:, k], color=c, lw=2, alpha=.45)
        ax[1].plot(t0, est[:, k], color=c, lw=1.2, ls="--", label=f"{lab} est")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("position [m]")
    ax[1].set_title("per-axis (solid = truth)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ax[2].plot(t0, err, color="#b3261e", lw=1.4)
    ax[2].axhline(np.sqrt(np.mean(err ** 2)), ls="--", c="k", lw=1,
                  label=f"RMSE {np.sqrt(np.mean(err**2)):.3f} m")
    ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("position error [m]")
    ax[2].set_title("error vs ground truth"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"wrote {out_png}")


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from ..evaluate import aligned_pair, evaluate

    rclpy.init(args=args)
    node = Node("tello_evaluate_bag")
    node.declare_parameter("bag", "")
    node.declare_parameter("est_topic", "/tello_vio/odom")
    node.declare_parameter("gt_topic", "/tello_ground_truth/pose")
    node.declare_parameter("plot", "trajectory_error.png")
    node.declare_parameter("rpe_window_s", 1.0)
    node.declare_parameter("fit_scale", False)

    bag = str(node.get_parameter("bag").value)
    est_topic = str(node.get_parameter("est_topic").value)
    gt_topic = str(node.get_parameter("gt_topic").value)
    out_png = str(node.get_parameter("plot").value)
    win = float(node.get_parameter("rpe_window_s").value)
    fit_scale = bool(node.get_parameter("fit_scale").value)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    if not bag:
        raise SystemExit("set -p bag:=<path to rosbag2 directory>")

    data = read_bag(bag, est_topic, gt_topic)
    t_est, p_est = data[est_topic]
    t_gt, p_gt = data[gt_topic]
    print(f"read {len(t_est)} estimate and {len(t_gt)} ground-truth samples")

    r = evaluate(t_est, p_est, t_gt, p_gt, rpe_window_s=win)
    print()
    print(r.report())
    print()
    if r.ate_sim3_rmse < 0.5 * r.ate_se3_rmse:
        print("NOTE: Sim3 error is much lower than SE3 -- the trajectory SHAPE is")
        print("      good but the SCALE is off by "
              f"{abs(1.0 - r.fitted_scale)*100:.0f} %. Look at calibration")
        print("      (speed_to_mps, camera intrinsics), not at the filter.")

    t, est, gt = aligned_pair(t_est, p_est, t_gt, p_gt, with_scale=fit_scale)
    err = np.linalg.norm(est - gt, axis=1)
    plot(t, est, gt, err, out_png,
         f"Tello VIO vs ArUco ground truth  |  ATE {r.ate_se3_rmse:.3f} m "
         f"({r.drift_percent:.1f} % of {r.path_length_m:.1f} m flown)")


if __name__ == "__main__":
    main()
