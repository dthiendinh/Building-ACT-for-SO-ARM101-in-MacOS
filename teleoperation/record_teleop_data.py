#!/usr/bin/env python3
"""Leader -> real follower teleop, dual camera feed + episode recording — LeRobot version.

Cùng logic với record_teleop_data.py gốc, nhưng thay toàn bộ phần serial protocol thô
(0x55 0x55 packets) và cv2.VideoCapture thủ công bằng lerobot.robots.so101_follower /
lerobot.teleoperators.so101_leader. Camera front + wrist được khai báo trực tiếp trong
config của robot (LeRobot hỗ trợ nhiều camera cùng lúc qua dict `cameras`), robot tự
connect/đọc cả 2 mỗi lần gọi get_observation() -- không cần tự quản lý VideoCapture nữa.

Vẫn giữ 2 process riêng (script này + sim_view_follower.py chạy qua Isaac Lab) vì lý do
tương tự bản gốc: chỉ 1 process được giữ port serial của follower tại 1 thời điểm, và
Kit UI / cv2 Qt window không share được 1 process. State được truyền qua follower_state.json.

KHÁC BIỆT CẦN LƯU Ý so với bản gốc:
  - Không còn tham số --tracking-ms: SO101Follower.send_action() ghi thẳng goal position,
    không có khái niệm "tracking_time_ms" per-call như giao thức cũ. Độ mượt di chuyển giờ
    phụ thuộc P-gain/tốc độ nội tại của servo, không chỉnh được per-tick từ code này.
  - "state" ghi vào dataset giờ lấy trực tiếp từ get_observation() (đọc thật từ follower
    mỗi tick, cùng lúc với action/camera) thay vì poll nền ~10Hz như bản gốc -- SO101Follower
    đã tối ưu sẵn tốc độ đọc bus nên không cần tách thread poll riêng nữa. Nếu bạn thấy loop
    tụt xuống dưới ~25Hz khi test, báo mình để đưa lại poll thread như cũ.

Usage (venv của project, không cần Isaac Lab cho nửa này):
    ./venv/bin/python record_teleop_data.py
"""


""" The data contrainer structure
teleoperation/
├── record_teleop_data.py
├── follower_state.json
└── data/
    └── pick_place_front_view_v3/
        ├── episode_0000/
        │   ├── front.mp4
        │   ├── wrist.mp4
        │   └── data.npz
        │
        ├── episode_0001/
        │   ├── front.mp4
        │   ├── wrist.mp4
        │   └── data.npz
        │
        └── episode_0002/
            ├── front.mp4
            ├── wrist.mp4
            └── data.npz
"""

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import json
import numpy as np
import os

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so_leader import (
    SO101Leader,
    SO101LeaderConfig
)

from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig
)

parser = argparse.ArgumentParser(description="SO-ARM101 teleop data recorder (LeRobot).")
parser.add_argument(
    "--data-dir",
    type=str,
    default="pick_place_front_view_v3",
    help="Subfolder under data/ to save episodes into.",
)
args_cli = parser.parse_args()

print("=========================================")
print("   SO-ARM101 TELEOP DATA RECORDER (LeRobot)")
print("=========================================\n")

# --- Sửa 4 giá trị này theo port + id bạn đã calibrate ---
LEADER_PORT = "/dev/tty.usbmodem5B8E1128291"
FOLLOWER_PORT = "/dev/tty.usbmodem5B8E1131141"

LEADER_ID = "so101_leader"
FOLLOWER_ID = "so101_follower"

# --- Camera: sửa index_or_path theo máy bạn (xem camera_utils.py để lấy index ổn định) ---
CAMERA_FPS = 30
cameras_config = {
    "front": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=CAMERA_FPS),
    "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=CAMERA_FPS),
}

# Canonical ACT vector order. Keep this identical to act/low_level_control.py.
# Never sort these keys alphabetically: that changes the physical joint mapped
# to each model output.
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
JOINT_KEYS = [f"{name}.pos" for name in MOTOR_NAMES]

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data" / args_cli.data_dir
STATE_FILE = SCRIPT_DIR / "follower_state.json"
CONTROL_HZ = 30
CONTROL_DT = 1.0 / CONTROL_HZ

follower_config = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, cameras=cameras_config)
leader_config = SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID)

follower = SO101Follower(follower_config)
leader = SO101Leader(leader_config)


def write_follower_state(observation):
    """Ghi state hiện tại xuống JSON cho sim_view_follower.py đọc (process khác)."""
    tmp_path = STATE_FILE.with_suffix(".tmp")
    # Chỉ lấy các key dạng ".pos" (joint state), bỏ ảnh camera ra khỏi JSON
    joint_state = {k: v for k, v in observation.items() if k.endswith(".pos")}
    tmp_path.write_text(json.dumps({"t": time.time(), "state": joint_state}))
    os.replace(tmp_path, STATE_FILE)


def next_episode_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p for p in OUTPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    next_idx = max([int(p.name.split("_")[1]) for p in existing], default=-1) + 1
    return OUTPUT_DIR / f"episode_{next_idx:04d}"


class LoopRateMonitor:
    """Đo tốc độ loop thực tế (Hz), in ra mỗi ~1s, cảnh báo khi tụt dưới target.
    Dùng để verify loop có giữ được CONTROL_HZ mong muốn hay không -- đặc biệt quan trọng
    sau khi bỏ follower-poll-thread riêng (xem docstring đầu file) vì nếu get_observation()
    của LeRobot chậm hơn dự kiến, loop sẽ tụt Hz y hệt vấn đề bản gốc từng gặp."""

    def __init__(self, target_hz, warn_threshold_ratio=0.85, report_every_s=1.0):
        self.target_hz = target_hz
        self.warn_threshold_hz = target_hz * warn_threshold_ratio
        self.report_every_s = report_every_s
        self.tick_times = []          # timestamp mỗi tick trong cửa sổ report hiện tại
        self.window_start = time.time()
        self.worst_hz_ever = float("inf")
        self.latest_hz = 0.0

    def tick(self):
        now = time.time()
        self.tick_times.append(now)
        elapsed = now - self.window_start
        if elapsed >= self.report_every_s:
            n = len(self.tick_times)
            hz = n / elapsed
            self.latest_hz = hz
            self.worst_hz_ever = min(self.worst_hz_ever, hz)
            print(
                f"\r\033[2KRate: {hz:.1f}/{self.target_hz} Hz "
                f"| worst: {self.worst_hz_ever:.1f} Hz",
                end="",
                flush=True,
            )
            self.tick_times = []
            self.window_start = now


class EpisodeWriter:
    """Ghi video cho MỖI camera (front.mp4, wrist.mp4...) + data.npz cho action/state/timestamp.
    Encode chạy trên thread riêng như bản gốc, main loop chỉ queue.put() không blocking."""

    def __init__(self, out_dir, fps, camera_names, frame_size, joint_names):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True)
        self.camera_names = camera_names
        self.joint_names = list(joint_names)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.videos = {
            name: cv2.VideoWriter(str(out_dir / f"{name}.mp4"), fourcc, fps, frame_size)
            for name in camera_names
        }
        self.timestamps = []
        self.actions = []
        self.states = []
        self.start_time = time.time()
        self.max_queue_size = 0
        self.frame_queue = queue.Queue()
        """
        Main thread:
        - Control Robot
        - Read Camera
        - Put the frames into the queue
        Writer thread:
        - Take the frames from the queue
        - Encode and record video
        """
        self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.writer_thread.start()

    def _write_loop(self):
        while True:
            item = self.frame_queue.get()
            if item is None:  # sentinel
                break
            frames_by_cam = item
            for name, frame in frames_by_cam.items():
                self.videos[name].write(frame)

    def append(self, timestamp, action, state, frames_by_cam):
        self.timestamps.append(timestamp)
        self.actions.append(action)
        self.states.append(state)
        self.frame_queue.put(frames_by_cam)
        self.max_queue_size = max(self.max_queue_size, self.frame_queue.qsize())

    def _finish_writer(self):
        self.frame_queue.put(None)
        self.writer_thread.join()
        for v in self.videos.values():
            v.release()

    def save(self):
        self._finish_writer()
        np.savez_compressed(
            self.out_dir / "data.npz",
            timestamps=np.array(self.timestamps, dtype=np.float64),
            actions=np.array(self.actions, dtype=np.float32),
            states=np.array(self.states, dtype=np.float32),
            joint_names=np.array(self.joint_names),
        )
        print(f"\n[SAVED] {len(self.timestamps)} steps -> {self.out_dir}/")

    def discard(self):
        self._finish_writer()
        import shutil
        shutil.rmtree(self.out_dir)
        print(f"\n[DISCARDED] {self.out_dir}/")




class DashboardRenderer:
    """OpenCV dashboard for live data collection.

    Pure rendering only: no extra threads and no hardware I/O. This keeps the
    control/recording path unchanged while making timing, tracking and writer
    health visible during collection.
    """

    WIDTH = 1280
    HEIGHT = 720
    HEADER_H = 72
    FOOTER_H = 46
    SIDE_W = 390
    PAD = 14

    BG = (18, 20, 24)
    PANEL = (29, 32, 38)
    PANEL_2 = (36, 40, 47)
    TEXT = (232, 235, 239)
    MUTED = (148, 155, 166)
    GREEN = (92, 205, 134)
    CYAN = (220, 190, 70)
    YELLOW = (72, 190, 245)
    RED = (78, 78, 238)

    def __init__(self, target_hz, joint_names):
        self.target_hz = target_hz
        self.joint_names = joint_names or []
        self.events = []

    def set_joint_names(self, names):
        self.joint_names = list(names)

    def log(self, message, level="INFO"):
        self.events.append((time.strftime("%H:%M:%S"), level, message))
        self.events = self.events[-5:]

    @staticmethod
    def _put(img, text, xy, scale=0.48, color=(232, 235, 239), thickness=1):
        cv2.putText(img, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _fit(frame, width, height):
        if frame is None or frame.size == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        h, w = frame.shape[:2]
        scale = min(width / w, height / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x = (width - nw) // 2
        y = (height - nh) // 2
        canvas[y:y + nh, x:x + nw] = resized
        return canvas

    def _metric(self, img, x, y, label, value, color=None):
        color = color or self.TEXT
        self._put(img, label.upper(), (x, y), 0.36, self.MUTED, 1)
        self._put(img, value, (x, y + 24), 0.60, color, 1)

    def _draw_joint_rows(self, img, x, y, w, state_vec, action_vec):
        self._put(img, "JOINT TRACKING", (x, y), 0.43, self.MUTED, 1)
        y += 22
        if not state_vec or not action_vec:
            self._put(img, "Waiting for robot state...", (x, y + 20), 0.45, self.MUTED, 1)
            return

        n = min(len(state_vec), len(action_vec), 6)
        row_h = 43
        bar_x = x + 105
        bar_w = max(80, w - 195)

        diffs = np.abs(np.asarray(action_vec[:n], dtype=float) - np.asarray(state_vec[:n], dtype=float))
        max_diff = float(np.max(diffs)) if len(diffs) else 0.0
        dynamic_scale = max(10.0, float(np.max(np.abs(action_vec[:n]))) if n else 10.0)

        for i in range(n):
            yy = y + i * row_h
            raw_name = self.joint_names[i] if i < len(self.joint_names) else f"joint_{i + 1}"
            name = raw_name.replace(".pos", "")[:13]
            st = float(state_vec[i])
            ac = float(action_vec[i])
            diff = abs(ac - st)
            diff_color = self.GREEN if diff < 4 else self.YELLOW if diff < 10 else self.RED

            self._put(img, name, (x, yy + 16), 0.39, self.TEXT, 1)
            cv2.rectangle(img, (bar_x, yy + 5), (bar_x + bar_w, yy + 13), self.PANEL_2, -1)
            center = bar_x + bar_w // 2
            cv2.line(img, (center, yy + 3), (center, yy + 15), self.MUTED, 1)

            def px(v):
                norm = np.clip(v / dynamic_scale, -1.0, 1.0)
                return int(center + norm * (bar_w / 2 - 2))

            sx, ax = px(st), px(ac)
            cv2.circle(img, (sx, yy + 9), 4, self.GREEN, -1, cv2.LINE_AA)
            cv2.circle(img, (ax, yy + 9), 4, self.CYAN, -1, cv2.LINE_AA)
            self._put(img, f"{st:6.1f} / {ac:6.1f}", (bar_x + bar_w + 10, yy + 14), 0.34, self.TEXT, 1)
            self._put(img, f"d {diff:4.1f}", (bar_x + bar_w + 10, yy + 30), 0.31, diff_color, 1)

        err_color = self.GREEN if max_diff < 4 else self.YELLOW if max_diff < 10 else self.RED
        self._put(img, f"max |action-state| = {max_diff:.2f}", (x, y + n * row_h + 5), 0.37, err_color, 1)

    def render(self, frames_by_cam, episode, rate_monitor, state_vec, action_vec):
        img = np.full((self.HEIGHT, self.WIDTH, 3), self.BG, dtype=np.uint8)
        recording = episode is not None
        status_color = self.RED if recording else self.MUTED

        cv2.rectangle(img, (0, 0), (self.WIDTH, self.HEADER_H), self.PANEL, -1)
        cv2.circle(img, (28, 28), 7, status_color, -1, cv2.LINE_AA)
        self._put(img, "SO-ARM101 / DATA CAPTURE", (48, 34), 0.72, self.TEXT, 2)
        self._put(img, "REC" if recording else "IDLE", (48, 57), 0.37, status_color, 1)

        hz = rate_monitor.latest_hz
        hz_color = self.GREEN if hz == 0 or hz >= self.target_hz * 0.85 else self.YELLOW if hz >= self.target_hz * 0.65 else self.RED
        self._metric(img, 520, 23, "Control loop", f"{hz:4.1f} / {self.target_hz} Hz", hz_color)

        frame_count = len(episode.timestamps) if recording else 0
        duration = time.time() - episode.start_time if recording else 0.0
        ep_name = episode.out_dir.name if recording else "--"
        self._metric(img, 740, 23, "Episode", ep_name, self.RED if recording else self.TEXT)
        self._metric(img, 910, 23, "Frames", str(frame_count), self.TEXT)
        self._metric(img, 1030, 23, "Time", f"{duration:05.1f}s", self.TEXT)
        self._metric(img, 1150, 23, "Cams", str(len(frames_by_cam)), self.TEXT)

        content_y = self.HEADER_H + self.PAD
        content_h = self.HEIGHT - self.HEADER_H - self.FOOTER_H - 2 * self.PAD
        cam_x = self.PAD
        cam_y = content_y
        cam_w = self.WIDTH - self.SIDE_W - 3 * self.PAD
        cam_h = content_h
        cv2.rectangle(img, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), self.PANEL, -1)

        cam_items = list(frames_by_cam.items())
        if len(cam_items) <= 1:
            slots = [(cam_x + 8, cam_y + 8, cam_w - 16, cam_h - 16)]
        else:
            gap = 8
            each_h = (cam_h - 24 - gap) // 2
            slots = [
                (cam_x + 8, cam_y + 8, cam_w - 16, each_h),
                (cam_x + 8, cam_y + 8 + each_h + gap, cam_w - 16, each_h),
            ]

        for (name, frame), (sx, sy, sw, sh) in zip(cam_items[:2], slots):
            view = self._fit(frame, sw, sh)
            img[sy:sy + sh, sx:sx + sw] = view
            cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), self.PANEL_2, 1)
            cv2.rectangle(img, (sx + 10, sy + 10), (sx + 115, sy + 36), self.BG, -1)
            self._put(img, f"CAM / {name.upper()}", (sx + 18, sy + 29), 0.40, self.TEXT, 1)

        side_x = cam_x + cam_w + self.PAD
        side_y = cam_y
        side_w = self.SIDE_W
        side_h = cam_h
        cv2.rectangle(img, (side_x, side_y), (side_x + side_w, side_y + side_h), self.PANEL, -1)
        self._draw_joint_rows(img, side_x + 18, side_y + 29, side_w - 36, state_vec, action_vec)

        queue_size = episode.frame_queue.qsize() if recording else 0
        q_color = self.GREEN if queue_size < 5 else self.YELLOW if queue_size < 15 else self.RED
        health_y = side_y + side_h - 158
        cv2.line(img, (side_x + 18, health_y - 18), (side_x + side_w - 18, health_y - 18), self.PANEL_2, 1)
        self._put(img, "SYSTEM HEALTH", (side_x + 18, health_y), 0.43, self.MUTED, 1)
        self._put(img, f"writer queue  {queue_size:>3}", (side_x + 18, health_y + 27), 0.40, q_color, 1)
        worst = rate_monitor.worst_hz_ever
        worst_text = "--" if worst == float("inf") else f"{worst:.1f} Hz"
        self._put(img, f"worst loop    {worst_text}", (side_x + 18, health_y + 50), 0.40, self.TEXT, 1)
        if recording:
            self._put(img, f"queue max     {episode.max_queue_size}", (side_x + 18, health_y + 73), 0.40, self.TEXT, 1)

        self._put(img, "EVENTS", (side_x + 18, health_y + 103), 0.38, self.MUTED, 1)
        last = self.events[-1] if self.events else ("--:--:--", "INFO", "Ready")
        tstamp, level, msg = last
        level_color = self.RED if level == "ERROR" else self.YELLOW if level == "WARN" else self.GREEN
        self._put(img, f"{tstamp} {level}", (side_x + 18, health_y + 126), 0.34, level_color, 1)
        self._put(img, msg[:43], (side_x + 18, health_y + 147), 0.33, self.TEXT, 1)

        fy = self.HEIGHT - self.FOOTER_H
        cv2.rectangle(img, (0, fy), (self.WIDTH, self.HEIGHT), self.PANEL, -1)
        self._put(img, "R  record", (20, fy + 29), 0.43, self.TEXT, 1)
        self._put(img, "S  save", (135, fy + 29), 0.43, self.TEXT, 1)
        self._put(img, "X  discard", (235, fy + 29), 0.43, self.TEXT, 1)
        self._put(img, "Q  quit", (365, fy + 29), 0.43, self.TEXT, 1)
        mode_text = "RECORDING DATA" if recording else "TELEOP / NOT RECORDING"
        self._put(img, mode_text, (self.WIDTH - 245, fy + 29), 0.42, status_color, 1)

        return img

try:
    follower.connect()
    print(f"[OK] Follower + {len(cameras_config)} camera(s) connected ({FOLLOWER_PORT})")
    leader.connect()
    print(f"[OK] Leader connected ({LEADER_PORT})")
except Exception as e:
    print(f"[ERROR] Connection failure: {e}")
    raise SystemExit(1)

print("\n[LIVE] Teleoperation running in camera window.")
print("  r = start recording a new episode")
print("  s = stop & save the current episode")
print("  x = discard the current episode")
print("  q = quit")
print(f"\n[INFO] Follower state -> {STATE_FILE} (for sim_view_follower.py in another terminal)\n")

episode = None
camera_names = list(cameras_config.keys())
joint_keys = JOINT_KEYS
rate_monitor = LoopRateMonitor(target_hz=CONTROL_HZ)
dashboard = DashboardRenderer(target_hz=CONTROL_HZ, joint_names=[])

try:
    while True:
        loop_start = time.time()

        # get_observation() trả về dict gồm joint state (key "*.pos") + 1 numpy frame
        # cho MỖI camera trong cameras_config (key = tên camera, ví dụ "front", "wrist")
        observation = follower.get_observation()
        action = leader.get_action()
        follower.send_action(action)

        missing_action_keys = [key for key in joint_keys if key not in action]
        missing_state_keys = [key for key in joint_keys if key not in observation]
        if missing_action_keys or missing_state_keys:
            raise KeyError(
                f"SO-ARM101 joint schema mismatch: "
                f"missing action keys={missing_action_keys}, "
                f"missing state keys={missing_state_keys}"
            )

        if dashboard.joint_names != joint_keys:
            dashboard.set_joint_names(joint_keys)
            dashboard.log("Canonical joint schema initialized")

        action_vec = [action[k] for k in joint_keys]
        state_vec = [observation[k] for k in joint_keys]

        frames_by_cam = {}
        for name in camera_names:
            frame = observation[name]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # LeRobot trả RGB, cv2 cần BGR
            # frame = cv2.rotate(frame, cv2.ROTATE_180)
            frames_by_cam[name] = frame

        write_follower_state(observation)

        if episode is not None:
            episode.append(loop_start, action_vec, state_vec, frames_by_cam)

        # display = frames_by_cam["wrist"].copy()
        # status = f"REC ({len(episode.timestamps)})" if episode is not None else "idle"
        # color = (0, 0, 255) if episode is not None else (200, 200, 200)
        # cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # cv2.imshow("SO-ARM101 Teleop Recorder", display)

        display = dashboard.render(
            frames_by_cam=frames_by_cam,
            episode=episode,
            rate_monitor=rate_monitor,
            state_vec=state_vec,
            action_vec=action_vec,
        )
        cv2.imshow("SO-ARM101 Data Capture Dashboard", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r") and episode is None:
            h, w = frames_by_cam["wrist"].shape[:2]
            episode = EpisodeWriter(
                next_episode_dir(), CAMERA_FPS, camera_names, (w, h), joint_keys
            )
            dashboard.log(f"Recording {episode.out_dir.name}", "INFO")
        elif key == ord("s") and episode is not None:
            saved_name = episode.out_dir.name
            episode.save()
            dashboard.log(f"Saved {saved_name}", "INFO")
            episode = None
        elif key == ord("x") and episode is not None:
            discarded_name = episode.out_dir.name
            episode.discard()
            dashboard.log(f"Discarded {discarded_name}", "WARN")
            episode = None
        elif key == ord("q"):
            break

        elapsed = time.time() - loop_start
        time.sleep(max(0, CONTROL_DT - elapsed))
        rate_monitor.tick()

except KeyboardInterrupt:
    print("\n[EXIT] Interrupted by user.")
except Exception as e:
    # Bắt riêng để phân biệt lỗi mất kết nối leader/follower, như bản teleop.py trước đó
    print(f"\n[ERROR] {e}")
finally:
    print()
    if episode is not None:
        episode.save()
    print("[SAFETY] Disconnecting...")
    try:
        follower.disconnect()
    except Exception as e:
        print(f"[WARN] Follower disconnect issue: {e}")
    try:
        leader.disconnect()
    except Exception as e:
        print(f"[WARN] Leader disconnect issue: {e}")
    cv2.destroyAllWindows()
    print("[DONE] Hardware connections safely released.")
