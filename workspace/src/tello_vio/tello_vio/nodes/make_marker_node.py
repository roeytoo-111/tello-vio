#!/usr/bin/env python3
"""Write a printable ArUco marker PNG.

    ros2 run tello_vio make_marker --ros-args -p output:=marker.png -p id:=0

Print at 100 % scale (disable "fit to page"), then MEASURE the black square
with a ruler and pass that to ground_truth as marker_size_m. The printed size
sets the scale of every ground-truth number, and printers routinely scale by a
few percent.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from ..fiducial import DEFAULT_DICT, generate_marker_png


def main(args=None):
    rclpy.init(args=args)
    node = Node("tello_make_marker")
    node.declare_parameter("output", "marker.png")
    node.declare_parameter("id", 0)
    node.declare_parameter("dictionary", DEFAULT_DICT)
    node.declare_parameter("pixels", 1200)

    out = str(node.get_parameter("output").value)
    mid = int(node.get_parameter("id").value)
    generate_marker_png(out, marker_id=mid,
                        px=int(node.get_parameter("pixels").value),
                        dict_name=str(node.get_parameter("dictionary").value))
    node.get_logger().info(
        f"wrote {out} (id {mid}). Print at 100 % scale, then MEASURE the black "
        "square and pass it as marker_size_m.")
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
