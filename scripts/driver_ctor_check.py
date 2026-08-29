"""Construct TelloNode against a fake drone, to catch init-time errors offline.

Run as a SUBPROCESS by test_static_analysis.py, not imported: rclpy.init() is
process-global and cannot be safely torn down and re-run inside a pytest
session shared with other tests.

This exists because `np.array(...)` in a module that imports `numpy` (no alias)
passed py_compile, built, installed, and only failed at construction time -- on
the drone, mid-session. Constructing the node against a stub driver exercises
every line of __init__ without hardware.
"""
import sys, types
from unittest import mock

fake = types.ModuleType("djitellopy")
class FakeTello:
    RESPONSE_TIMEOUT = 7
    TELLO_IP = "192.168.10.1"
    def __init__(self, host=None, **kw): self.retry_count = 3
    def connect(self): pass
    def streamon(self): pass
    def get_frame_read(self, with_queue=False): return mock.MagicMock(frame=None)
    def get_current_state(self): return {}
    def send_rc_control(self, *a): pass
    def __getattr__(self, n): return lambda *a, **k: None
fake.Tello = FakeTello
sys.modules["djitellopy"] = fake

import tello.node as tn
tn.check_network_path = lambda ip: None      # bypass the pre-flight net check

import rclpy
rclpy.init()
node = rclpy.create_node("ctor_test")
try:
    drone = tn.TelloNode(node)
    print("TelloNode constructed OK")
    print("  mission_pad_signs =", drone.mission_pad_signs)
    st = {"mid": 3, "x": 50, "y": 30, "z": 80, "mpry": "1,2,3"}
    drone._publish_mission_pad(node.get_clock().now().to_msg(), st)
    print("  _publish_mission_pad OK (pad detected)")
    drone._publish_mission_pad(node.get_clock().now().to_msg(), {})
    print("  _publish_mission_pad OK (no pad)")
    drone._shutdown.set()
finally:
    node.destroy_node(); rclpy.shutdown()

