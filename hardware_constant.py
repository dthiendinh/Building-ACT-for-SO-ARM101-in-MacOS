"""Shared hardware settings for teleoperation, recording and evaluation."""
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Discover each arm with lerobot-find-port; use the same IDs for calibration.
LEADER_PORT = "/dev/tty.usbmodem5B8E1128291"
FOLLOWER_PORT = "/dev/tty.usbmodem5B8E1131141"
LEADER_ID = "so101_leader"
FOLLOWER_ID = "so101_follower"

CAMERA_FPS = 30
CAMERA_CONFIGS = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=CAMERA_FPS),
    "wrist": OpenCVCameraConfig(index_or_path=1, width=1280, height=720, fps=CAMERA_FPS),
}
CONTROL_HZ = 30.0
CONTROL_DT = 1.0 / CONTROL_HZ

# Joint order shared by the recorder, dataset loader and robot environment.
MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
JOINT_KEYS = [f"{name}.pos" for name in MOTOR_NAMES]
