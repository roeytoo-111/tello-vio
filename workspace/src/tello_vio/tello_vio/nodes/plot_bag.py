#!/usr/bin/env python3
"""Plot the flown trajectory. No ground truth required.

``evaluate_bag`` scores VIO *against* a fiducial ground-truth topic. This node
answers the more basic question -- "what path does the filter think it flew?"
-- from the estimate alone, so it works on any recorded flight.

    # live: fly, then Ctrl-C to write the figure
    ros2 run tello_vio plot_bag --ros-args -p plot:=flight.png

    # from a bag
    ros2 bag record -o flight1 /tello_vio/odom /tello/odom /tello/tof
    ros2 run tello_vio plot_bag --ros-args -p bag:=flight1 -p plot:=flight.png

Six panels: top-down path, 3D path, per-axis position with the filter's own
1-sigma envelope, height against the independent ToF and barometer signals,
speed, and how the uncertainty grows. Overlaying the driver's telemetry-only
dead reckoning (``/tello/odom``) shows what vision is actually contributing.
"""

from __future__ import annotations

import numpy as np

# Diagonal indices of a row-major 6x6 covariance: x, y, z.
_COV_XYZ = (0, 7, 14)


class Track:
    """Timestamped poses, twists and position sigmas from one topic."""

    def __init__(self) -> None:
        self.t: list[float] = []
        self.p: list[list[float]] = []
        self.v: list[float] = []
        self.s: list[list[float]] = []

    def add(self, t: float, p, v=None, cov=None) -> None:
        self.t.append(t)
        self.p.append([p.x, p.y, p.z])
        self.v.append(float(np.linalg.norm([v.x, v.y, v.z])) if v is not None
                      else float("nan"))
        self.s.append([float(np.sqrt(max(cov[i], 0.0))) for i in _COV_XYZ]
                      if cov is not None else [float("nan")] * 3)

    def finish(self):
        """Freeze the accumulated lists into arrays."""
        self.t = np.asarray(self.t, float)
        self.p = np.asarray(self.p, float).reshape(-1, 3)
        self.v = np.asarray(self.v, float)
        self.s = np.asarray(self.s, float).reshape(-1, 3)
        return self

    def __len__(self) -> int:
        return len(self.t)


def _ingest(track: Track, msg) -> None:
    """Accept Odometry, PoseStamped or Range into a Track."""
    h = msg.header
    t = h.stamp.sec + h.stamp.nanosec * 1e-9
    if hasattr(msg, "range"):                       # sensor_msgs/Range
        track.t.append(t)
        track.p.append([float("nan"), float("nan"), float(msg.range)])
        track.v.append(float("nan"))
        track.s.append([float("nan")] * 3)
        return
    pose = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
    twist = msg.twist.twist.linear if hasattr(msg, "twist") else None
    cov = msg.pose.covariance if hasattr(msg.pose, "covariance") else None
    track.add(t, pose.position, twist, cov)


def read_bag(path: str, topics: list[str]) -> dict[str, Track]:
    """Read the requested topics out of a rosbag2. Missing topics are skipped."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = [t for t in topics if t in types]
    if not wanted:
        raise SystemExit(
            f"none of {topics} are in the bag. Recorded topics:\n  " +
            "\n  ".join(sorted(types)))
    for missing in (t for t in topics if t not in types):
        print(f"note: {missing} not in bag -- that panel will be empty")

    out = {t: Track() for t in wanted}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in out:
            _ingest(out[topic], deserialize_message(data, get_message(types[topic])))
    return {k: v.finish() for k, v in out.items()}


def read_live(topics: list[str], est_topic: str) -> dict[str, Track]:
    """Subscribe until Ctrl-C, then return what arrived."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Range

    tracks = {t: Track() for t in topics}
    node = Node("tello_plot_live")
    for topic in topics:
        msg_type = Range if topic.endswith("tof") else Odometry
        node.create_subscription(
            msg_type, topic,
            lambda m, tr=tracks[topic]: _ingest(tr, m),
            qos_profile_sensor_data)
    print(f"recording {est_topic} ... fly, then press Ctrl-C to plot")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print()
    node.destroy_node()
    return {k: v.finish() for k, v in tracks.items() if len(v)}


def plot(est: Track, cmp_: Track | None, tof: Track | None,
         out_png: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = est.t - est.t[0]
    fig = plt.figure(figsize=(16.5, 9.0))
    fig.suptitle(title, fontsize=13, weight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.26)
    axes = ["xy", "3d", "axis", "height", "speed", "sigma"]
    ax, have_3d = {}, True
    for i, k in enumerate(axes):
        cell = gs[i // 3, i % 3]
        if k != "3d":
            ax[k] = fig.add_subplot(cell)
            continue
        # Axes3D is unavailable when matplotlib is installed twice (system
        # package plus pip). Fall back to a side elevation rather than crash.
        try:
            ax[k] = fig.add_subplot(cell, projection="3d")
        except ValueError:
            have_3d = False
            ax[k] = fig.add_subplot(cell)

    # --- top-down, coloured by time so direction of travel is readable ------
    a = ax["xy"]
    a.scatter(est.p[:, 0], est.p[:, 1], c=t0, cmap="viridis", s=6, zorder=3)
    if cmp_ is not None and len(cmp_):
        a.plot(cmp_.p[:, 0], cmp_.p[:, 1], color="#999999", lw=1.2, ls="--",
               zorder=2, label="telemetry only")
        a.legend(fontsize=8, loc="best")
    a.scatter(*est.p[0, :2], c="green", s=90, marker="o", zorder=5, ec="k")
    a.scatter(*est.p[-1, :2], c="red", s=90, marker="X", zorder=5, ec="k")
    a.set_xlabel("x forward [m]"); a.set_ylabel("y left [m]")
    a.set_title("top-down (green = start, red = end, colour = time)")
    a.axis("equal"); a.grid(alpha=.3)

    # --- 3d, or a side elevation if this matplotlib has no Axes3D ----------
    a = ax["3d"]
    if have_3d:
        a.plot(est.p[:, 0], est.p[:, 1], est.p[:, 2], color="#1f6feb", lw=1.4)
        a.scatter(*est.p[0], c="green", s=60)
        a.scatter(*est.p[-1], c="red", s=60, marker="X")
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_zlabel("z up [m]")
        a.set_title("3D path")
    else:
        a.plot(est.p[:, 0], est.p[:, 2], color="#1f6feb", lw=1.4)
        a.scatter(est.p[0, 0], est.p[0, 2], c="green", s=70, zorder=5, ec="k")
        a.scatter(est.p[-1, 0], est.p[-1, 2], c="red", s=70, marker="X",
                  zorder=5, ec="k")
        a.set_xlabel("x forward [m]"); a.set_ylabel("z up [m]")
        a.set_title("side elevation (no Axes3D in this matplotlib)")
        a.axis("equal"); a.grid(alpha=.3)

    # --- per axis, with the filter's own 1-sigma envelope -------------------
    a = ax["axis"]
    for k, (lab, c) in enumerate(zip("xyz", ["#1f6feb", "#c2600f", "#1f8a4c"])):
        a.plot(t0, est.p[:, k], color=c, lw=1.4, label=lab)
        if np.isfinite(est.s[:, k]).any():
            a.fill_between(t0, est.p[:, k] - est.s[:, k],
                           est.p[:, k] + est.s[:, k], color=c, alpha=.18, lw=0)
    a.set_xlabel("time [s]"); a.set_ylabel("position [m]")
    a.set_title("per-axis position, shaded $\\pm 1\\sigma$")
    a.legend(fontsize=8); a.grid(alpha=.3)

    # --- height against the independent sensors ----------------------------
    a = ax["height"]
    a.plot(t0, est.p[:, 2], color="#1f6feb", lw=1.6, label="VIO z")
    if tof is not None and len(tof):
        a.plot(tof.t - est.t[0], tof.p[:, 2], color="#b3261e", lw=1.0,
               alpha=.75, label="ToF range")
    if cmp_ is not None and len(cmp_):
        a.plot(cmp_.t - est.t[0], cmp_.p[:, 2], color="#999999", lw=1.0,
               ls="--", label="barometer height")
    a.set_xlabel("time [s]"); a.set_ylabel("height [m]")
    a.set_title("height vs independent sensors")
    a.legend(fontsize=8); a.grid(alpha=.3)

    # --- speed: the panel that makes a runaway obvious ---------------------
    a = ax["speed"]
    if np.isfinite(est.v).any():
        a.plot(t0, est.v, color="#6f42c1", lw=1.3)
        vmax = float(np.nanmax(est.v))
        # Only draw the reference line when it is near the data, otherwise it
        # rescales the axis and flattens the signal we came here to read.
        if vmax > 1.0:
            a.axhline(2.0, ls="--", c="#b3261e", lw=1,
                      label="2 m/s -- implausible indoors")
            a.legend(fontsize=8)
        a.set_ylim(0, max(vmax * 1.15, 0.05))
    a.set_xlabel("time [s]"); a.set_ylabel("speed [m/s]")
    a.set_title("speed magnitude"); a.grid(alpha=.3)

    # --- uncertainty growth ------------------------------------------------
    a = ax["sigma"]
    for k, (lab, c) in enumerate(zip("xyz", ["#1f6feb", "#c2600f", "#1f8a4c"])):
        if np.isfinite(est.s[:, k]).any():
            a.plot(t0, est.s[:, k], color=c, lw=1.3, label=f"$\\sigma_{lab}$")
    a.set_xlabel("time [s]"); a.set_ylabel("1$\\sigma$ [m]")
    a.set_title("filter's own position uncertainty")
    a.legend(fontsize=8); a.grid(alpha=.3)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def summarise(est: Track) -> str:
    """Numbers you can read without a ground-truth reference."""
    dur = est.t[-1] - est.t[0]
    step = np.linalg.norm(np.diff(est.p, axis=0), axis=1)
    length = float(step.sum())
    net = float(np.linalg.norm(est.p[-1] - est.p[0]))
    lines = [
        f"samples          {len(est)}  over {dur:.1f} s  "
        f"({len(est) / dur:.1f} Hz)" if dur > 0 else f"samples {len(est)}",
        f"path length      {length:.2f} m",
        f"net displacement {net:.2f} m   "
        f"(end - start = {np.round(est.p[-1] - est.p[0], 2)})",
        f"height range     {np.nanmin(est.p[:, 2]):.2f} .. "
        f"{np.nanmax(est.p[:, 2]):.2f} m",
    ]
    if np.isfinite(est.v).any():
        lines.append(f"speed            mean {np.nanmean(est.v):.2f}  "
                     f"max {np.nanmax(est.v):.2f} m/s")
    if np.isfinite(est.s).any():
        lines.append(f"final 1-sigma    {np.round(est.s[-1], 3)} m")
    lines.append("")
    lines.append("If you took off and landed on the same spot, net displacement")
    lines.append("IS the accumulated drift -- the one honest error number you")
    lines.append("can get without a ground-truth reference.")
    if np.isfinite(est.v).any() and np.nanmax(est.v) > 2.0:
        lines.append("")
        lines.append(f"WARNING: peak speed {np.nanmax(est.v):.2f} m/s. Indoors that is")
        lines.append("         not real -- check speed_to_mps (Section 12 of the report).")
    return "\n".join(lines)


def main(args=None):
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("tello_plot_bag")
    node.declare_parameter("bag", "")
    node.declare_parameter("est_topic", "/tello_vio/odom")
    node.declare_parameter("compare_topic", "/tello/odom")
    node.declare_parameter("tof_topic", "/tello/tof")
    node.declare_parameter("plot", "trajectory.png")

    bag = str(node.get_parameter("bag").value)
    est_topic = str(node.get_parameter("est_topic").value)
    cmp_topic = str(node.get_parameter("compare_topic").value)
    tof_topic = str(node.get_parameter("tof_topic").value)
    out_png = str(node.get_parameter("plot").value)
    node.destroy_node()

    topics = [t for t in (est_topic, cmp_topic, tof_topic) if t]
    if bag:
        if rclpy.ok():
            rclpy.shutdown()
        data = read_bag(bag, topics)
    else:
        data = read_live(topics, est_topic)
        if rclpy.ok():
            rclpy.shutdown()

    est = data.get(est_topic)
    if est is None or len(est) < 2:
        raise SystemExit(
            f"no usable samples on {est_topic}. Is the VIO node running "
            "and publishing? Check `ros2 topic hz " + est_topic + "`.")

    print()
    print(summarise(est))
    print()
    plot(est, data.get(cmp_topic), data.get(tof_topic), out_png,
         f"Tello VIO trajectory  |  {len(est)} poses, "
         f"{est.t[-1] - est.t[0]:.0f} s, "
         f"{np.linalg.norm(np.diff(est.p, axis=0), axis=1).sum():.1f} m flown")


if __name__ == "__main__":
    main()
