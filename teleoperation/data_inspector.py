#!/usr/bin/env python3
"""
SO-ARM101 Dataset Inspector
===========================

Visual inspector + lightweight validator for the custom dataset produced by
record_teleop_data.py / so101_dashboard_recorder.py.

Expected dataset structure
--------------------------
data/
└── <dataset_name>/
    ├── episode_0000/
    │   ├── wrist.mp4
    │   ├── front.mp4        # optional
    │   └── data.npz
    ├── episode_0001/
    │   └── ...
    └── dataset_review.json  # created by this inspector

data.npz is expected to contain:
    timestamps : (N,)
    actions    : (N, J)
    states     : (N, J)

Controls
--------
SPACE          Play / pause
LEFT / RIGHT   Previous / next frame
J / L          -10 / +10 frames
A / D          Previous / next episode
1..9           Select joint for detailed plot
G              Tag episode GOOD
B              Tag episode BAD
C              Tag BAD_CAMERA
M              Tag BAD_MOTION
F              Tag BAD_TASK
U              Clear tag (UNREVIEWED)
S              Save review file now
Q / ESC        Quit

Mouse:
    Use the "Frame" trackbar to scrub through the episode.

Example
-------
python dataset_inspector.py --dataset-dir data/pick_place_front_view_v3
"""

import argparse
import json
import os
import time

import cv2
import numpy as np


WINDOW_NAME = "SO-ARM101 Dataset Inspector"

# OpenCV BGR colors
BG = (18, 20, 24)
PANEL = (28, 31, 37)
PANEL_2 = (35, 39, 46)
TEXT = (235, 238, 242)
MUTED = (145, 151, 162)
GRID = (65, 70, 80)
GREEN = (90, 210, 120)
YELLOW = (60, 210, 240)
RED = (75, 80, 245)
CYAN = (235, 190, 70)
ORANGE = (60, 150, 245)
WHITE = (255, 255, 255)

ACTION_COLOR = CYAN
STATE_COLOR = ORANGE
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(img, text, xy, scale=0.5, color=TEXT, thickness=1):
    cv2.putText(img, str(text), xy, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, 1)
    if title:
        cv2.rectangle(img, (x, y), (x + w, y + 28), PANEL_2, -1)
        put_text(img, title, (x + 10, y + 19), 0.48, TEXT, 1)


def fit_image(frame, width, height):
    if frame is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = PANEL
        put_text(canvas, "NO FRAME", (20, 40), 0.7, RED, 2)
        return canvas

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = min(width / w, height / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (12, 14, 17)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def draw_badge(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.48, 1)
    pad_x, pad_y = 8, 6
    cv2.rectangle(img, (x, y), (x + tw + 2 * pad_x, y + th + 2 * pad_y), color, -1)
    put_text(img, text, (x + pad_x, y + th + pad_y - 2), 0.48, (10, 10, 10), 1)
    return tw + 2 * pad_x


def discover_episodes(dataset_dir):
    if not os.path.isdir(dataset_dir):
        return []
    episodes = []
    for name in sorted(os.listdir(dataset_dir)):
        full = os.path.join(dataset_dir, name)
        if name.startswith("episode_") and os.path.isdir(full):
            episodes.append(full)
    return episodes


def load_review_file(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_review_file(path, review):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def natural_camera_order(names):
    preferred = ["front", "wrist"]
    ordered = [name for name in preferred if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


class Episode:
    def __init__(self, path, target_fps=30.0, max_action_jump=25.0, max_dt_factor=2.5):
        self.path = path
        self.name = os.path.basename(path)
        self.target_fps = float(target_fps)
        self.max_action_jump = float(max_action_jump)
        self.max_dt_factor = float(max_dt_factor)

        self.timestamps = np.array([], dtype=np.float64)
        self.actions = np.empty((0, 0), dtype=np.float32)
        self.states = np.empty((0, 0), dtype=np.float32)

        self.video_paths = {}
        self.captures = {}
        self.video_info = {}
        self.errors = []
        self.warnings = []
        self.metrics = {}

        self._load_npz()
        self._discover_videos()
        self._validate()

    @property
    def num_samples(self):
        return int(len(self.timestamps))

    @property
    def num_joints(self):
        return int(self.actions.shape[1]) if self.actions.ndim == 2 else 0

    @property
    def duration(self):
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def _load_npz(self):
        npz_path = os.path.join(self.path, "data.npz")
        if not os.path.exists(npz_path):
            self.errors.append("Missing data.npz")
            return
        try:
            with np.load(npz_path) as data:
                required = ["timestamps", "actions", "states"]
                missing = [k for k in required if k not in data]
                if missing:
                    self.errors.append("Missing array(s): " + ", ".join(missing))
                    return
                self.timestamps = np.asarray(data["timestamps"], dtype=np.float64)
                self.actions = np.asarray(data["actions"], dtype=np.float32)
                self.states = np.asarray(data["states"], dtype=np.float32)
        except Exception as e:
            self.errors.append(f"Cannot load data.npz: {e}")

    def _discover_videos(self):
        if not os.path.isdir(self.path):
            return
        for filename in sorted(os.listdir(self.path)):
            if not filename.lower().endswith(".mp4"):
                continue
            name = os.path.splitext(filename)[0]
            path = os.path.join(self.path, filename)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.warnings.append(f"Cannot open {filename}")
                continue
            self.video_paths[name] = path
            self.captures[name] = cap
            self.video_info[name] = {
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }

    def _validate(self):
        n_t, n_a, n_s = len(self.timestamps), len(self.actions), len(self.states)
        self.metrics.update({"n_timestamps": n_t, "n_actions": n_a, "n_states": n_s})

        if not (n_t == n_a == n_s):
            self.errors.append(f"Length mismatch: t={n_t}, action={n_a}, state={n_s}")

        if self.actions.ndim != 2:
            self.errors.append(f"actions should be 2D, got {self.actions.shape}")
        if self.states.ndim != 2:
            self.errors.append(f"states should be 2D, got {self.states.shape}")
        if self.actions.ndim == 2 and self.states.ndim == 2 and self.actions.shape != self.states.shape:
            self.errors.append(f"Action/state shape mismatch: {self.actions.shape} vs {self.states.shape}")

        if self.actions.size:
            if np.isnan(self.actions).any():
                self.errors.append("NaN found in actions")
            if np.isinf(self.actions).any():
                self.errors.append("Inf found in actions")
        if self.states.size:
            if np.isnan(self.states).any():
                self.errors.append("NaN found in states")
            if np.isinf(self.states).any():
                self.errors.append("Inf found in states")

        if len(self.timestamps) > 1:
            dt = np.diff(self.timestamps)
            self.metrics["mean_dt_ms"] = float(np.mean(dt) * 1000.0)
            self.metrics["median_dt_ms"] = float(np.median(dt) * 1000.0)
            self.metrics["max_dt_ms"] = float(np.max(dt) * 1000.0)

            positive = dt[dt > 0]
            self.metrics["mean_hz"] = float(1.0 / np.mean(positive)) if len(positive) else 0.0
            self.metrics["median_hz"] = float(1.0 / np.median(positive)) if len(positive) else 0.0

            if np.any(dt <= 0):
                self.errors.append("Timestamps are not strictly increasing")

            expected_dt = 1.0 / max(self.target_fps, 1e-6)
            gap_idx = np.where(dt > expected_dt * self.max_dt_factor)[0]
            self.metrics["large_timing_gaps"] = int(len(gap_idx))
            if len(gap_idx):
                self.warnings.append(f"{len(gap_idx)} large timing gap(s)")

        if self.actions.ndim == 2 and self.actions.shape[0] > 1:
            delta = np.abs(np.diff(self.actions, axis=0))
            self.metrics["max_action_jump"] = float(np.max(delta))
            jump_count = int(np.sum(delta > self.max_action_jump))
            self.metrics["flagged_action_jumps"] = jump_count
            if jump_count:
                self.warnings.append(f"{jump_count} action jump(s) > {self.max_action_jump:g}")

        if self.actions.shape == self.states.shape and self.actions.size:
            err = np.abs(self.actions - self.states)
            self.metrics["mean_tracking_error"] = float(np.mean(err))
            self.metrics["max_tracking_error"] = float(np.max(err))

        if not self.video_info:
            self.errors.append("No readable .mp4 video found")

        for name, info in self.video_info.items():
            diff = abs(info["frames"] - n_t)
            self.metrics[f"{name}_frame_diff"] = int(diff)
            if diff:
                self.warnings.append(f"{name}.mp4 differs from data by {diff} frame(s)")

    def status(self):
        if self.errors:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"

    def get_frame(self, camera_name, frame_index):
        cap = self.captures.get(camera_name)
        if cap is None:
            return None
        frame_index = max(0, int(frame_index))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        return frame if ok else None

    def close(self):
        for cap in self.captures.values():
            try:
                cap.release()
            except Exception:
                pass


class DatasetInspector:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = os.path.abspath(args.dataset_dir)
        self.episode_paths = discover_episodes(self.dataset_dir)
        if not self.episode_paths:
            raise RuntimeError(f"No episode_* folders found in: {self.dataset_dir}")

        self.review_path = os.path.join(self.dataset_dir, "dataset_review.json")
        self.review = load_review_file(self.review_path)

        self.episode_index = 0
        self.episode = None
        self.frame_index = 0
        self.playing = False
        self.selected_joint = 0
        self.last_tick = time.time()
        self.trackbar_internal_update = False
        self.canvas_w = args.width
        self.canvas_h = args.height

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.canvas_w, self.canvas_h)
        cv2.createTrackbar("Frame", WINDOW_NAME, 0, 1, self._trackbar_callback)
        self.load_episode(0)

    def _trackbar_callback(self, value):
        if self.trackbar_internal_update or self.episode is None:
            return
        self.frame_index = int(np.clip(value, 0, max(self.episode.num_samples - 1, 0)))
        self.playing = False

    def set_trackbar(self):
        if self.episode is None:
            return
        self.trackbar_internal_update = True
        cv2.setTrackbarPos("Frame", WINDOW_NAME, int(self.frame_index))
        self.trackbar_internal_update = False

    def load_episode(self, index):
        if self.episode is not None:
            self.episode.close()

        self.episode_index = int(np.clip(index, 0, len(self.episode_paths) - 1))
        self.episode = Episode(
            self.episode_paths[self.episode_index],
            target_fps=self.args.target_fps,
            max_action_jump=self.args.max_action_jump,
            max_dt_factor=self.args.max_dt_factor,
        )
        self.frame_index = 0
        self.playing = False
        self.selected_joint = min(self.selected_joint, max(self.episode.num_joints - 1, 0))

        try:
            cv2.setTrackbarMax("Frame", WINDOW_NAME, max(self.episode.num_samples - 1, 1))
        except Exception:
            pass
        self.set_trackbar()

    def get_review_tag(self):
        record = self.review.get(self.episode.name)
        if isinstance(record, dict):
            return record.get("tag", "UNREVIEWED")
        if isinstance(record, str):
            return record
        return "UNREVIEWED"

    def set_review_tag(self, tag):
        self.review[self.episode.name] = {
            "tag": tag,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "episode_index": self.episode_index,
            "num_samples": self.episode.num_samples,
            "duration_s": round(self.episode.duration, 3),
            "technical_status": self.episode.status(),
        }
        save_review_file(self.review_path, self.review)

    def step_frame(self, delta):
        self.playing = False
        self.frame_index = int(np.clip(self.frame_index + delta, 0, max(self.episode.num_samples - 1, 0)))
        self.set_trackbar()

    def change_episode(self, delta):
        new_index = self.episode_index + delta
        if 0 <= new_index < len(self.episode_paths):
            self.load_episode(new_index)

    def update_playback(self):
        if not self.playing:
            self.last_tick = time.time()
            return

        now = time.time()
        fps = self.args.target_fps
        if self.episode.video_info:
            first = next(iter(self.episode.video_info.values()))
            if first["fps"] > 1:
                fps = first["fps"]

        dt = 1.0 / max(fps, 1.0)
        if now - self.last_tick >= dt:
            steps = max(1, int((now - self.last_tick) / dt))
            self.frame_index += steps
            if self.frame_index >= self.episode.num_samples:
                self.frame_index = max(self.episode.num_samples - 1, 0)
                self.playing = False
            self.set_trackbar()
            self.last_tick = now

    def draw_header(self, canvas):
        h = 68
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, h), (15, 17, 20), -1)
        put_text(canvas, "SO-ARM101 / DATASET INSPECTOR", (22, 27), 0.68, TEXT, 2)
        put_text(
            canvas,
            f"EP {self.episode_index + 1}/{len(self.episode_paths)}  {self.episode.name}",
            (22, 53),
            0.48,
            MUTED,
            1,
        )

        status = self.episode.status()
        status_color = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}[status]
        x = self.canvas_w - 475
        x += draw_badge(canvas, f"TECH {status}", x, 17, status_color) + 10

        tag = self.get_review_tag()
        tag_color = GREEN if tag == "GOOD" else (RED if tag.startswith("BAD") else PANEL_2)
        draw_badge(canvas, tag, x, 17, tag_color)

    def draw_camera_area(self, canvas, x, y, w, h):
        draw_panel(canvas, x, y, w, h, "CAMERA REVIEW")
        names = natural_camera_order(list(self.episode.video_info.keys()))
        if not names:
            put_text(canvas, "No readable camera videos", (x + 20, y + 70), 0.6, RED, 2)
            return

        content_y = y + 35
        content_h = h - 43
        if len(names) == 1:
            regions = [(names[0], x + 8, content_y, w - 16, content_h)]
        else:
            gap = 8
            each_h = (content_h - gap) // 2
            regions = [
                (names[0], x + 8, content_y, w - 16, each_h),
                (names[1], x + 8, content_y + each_h + gap, w - 16, content_h - each_h - gap),
            ]

        for name, rx, ry, rw, rh in regions:
            frame = self.episode.get_frame(name, self.frame_index)
            view = fit_image(frame, rw, rh)
            canvas[ry:ry + rh, rx:rx + rw] = view
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), GRID, 1)
            badge_w = draw_badge(canvas, f"CAM / {name.upper()}", rx + 10, ry + 10, PANEL_2)
            info = self.episode.video_info[name]
            put_text(
                canvas,
                f"{info['width']}x{info['height']} {info['fps']:.1f} FPS",
                (rx + badge_w + 26, ry + 29),
                0.42,
                WHITE,
                1,
            )

    def _normalized_points(self, values, x, y, w, h, vmin, vmax):
        if len(values) < 2:
            return None
        span = max(vmax - vmin, 1e-6)
        xs = np.linspace(x, x + w, len(values))
        ys = y + h - ((values - vmin) / span) * h
        pts = np.stack([xs, ys], axis=1)
        pts = np.round(pts).astype(np.int32)
        return pts.reshape((-1, 1, 2))

    def draw_joint_plot(self, canvas, x, y, w, h):
        title = f"JOINT {self.selected_joint + 1} / {max(self.episode.num_joints, 1)}"
        draw_panel(canvas, x, y, w, h, title)

        if self.episode.num_joints <= 0 or self.episode.actions.size == 0 or self.episode.states.size == 0:
            put_text(canvas, "No valid state/action arrays", (x + 18, y + 65), 0.55, RED, 1)
            return

        j = int(np.clip(self.selected_joint, 0, self.episode.num_joints - 1))
        a = self.episode.actions[:, j]
        s = self.episode.states[:, j]
        n = min(len(a), len(s))
        if n == 0:
            return
        a, s = a[:n], s[:n]

        pad_left, pad_right, pad_top, pad_bottom = 46, 16, 52, 38
        px, py = x + pad_left, y + pad_top
        pw, ph = w - pad_left - pad_right, h - pad_top - pad_bottom

        combined = np.concatenate([a, s])
        vmin, vmax = float(np.nanmin(combined)), float(np.nanmax(combined))
        if abs(vmax - vmin) < 1e-6:
            vmin -= 1.0
            vmax += 1.0
        else:
            margin = 0.08 * (vmax - vmin)
            vmin -= margin
            vmax += margin

        for k in range(5):
            gy = int(py + k * ph / 4)
            cv2.line(canvas, (px, gy), (px + pw, gy), GRID, 1)
        for k in range(6):
            gx = int(px + k * pw / 5)
            cv2.line(canvas, (gx, py), (gx, py + ph), GRID, 1)

        put_text(canvas, f"{vmax:.1f}", (x + 5, py + 5), 0.38, MUTED, 1)
        put_text(canvas, f"{vmin:.1f}", (x + 5, py + ph), 0.38, MUTED, 1)

        a_pts = self._normalized_points(a, px, py, pw, ph, vmin, vmax)
        s_pts = self._normalized_points(s, px, py, pw, ph, vmin, vmax)
        if a_pts is not None:
            cv2.polylines(canvas, [a_pts], False, ACTION_COLOR, 2, cv2.LINE_AA)
        if s_pts is not None:
            cv2.polylines(canvas, [s_pts], False, STATE_COLOR, 2, cv2.LINE_AA)

        current = int(np.clip(self.frame_index, 0, max(n - 1, 0)))
        cursor_x = int(px + current / max(n - 1, 1) * pw)
        cv2.line(canvas, (cursor_x, py), (cursor_x, py + ph), WHITE, 1)

        action_now = float(a[current])
        state_now = float(s[current])
        error_now = abs(action_now - state_now)

        put_text(canvas, f"ACTION {action_now:8.2f}", (x + 16, y + h - 14), 0.46, ACTION_COLOR, 1)
        put_text(canvas, f"STATE {state_now:8.2f}", (x + 160, y + h - 14), 0.46, STATE_COLOR, 1)
        put_text(canvas, f"|A-S| {error_now:7.2f}", (x + 290, y + h - 14), 0.46, TEXT, 1)

        cv2.line(canvas, (x + 18, y + 42), (x + 42, y + 42), ACTION_COLOR, 2)
        put_text(canvas, "action", (x + 48, y + 46), 0.4, MUTED, 1)
        cv2.line(canvas, (x + 112, y + 42), (x + 136, y + 42), STATE_COLOR, 2)
        put_text(canvas, "state", (x + 142, y + 46), 0.4, MUTED, 1)

    def draw_tracking_panel(self, canvas, x, y, w, h):
        draw_panel(canvas, x, y, w, h, "CURRENT FRAME / TRACKING")
        ep = self.episode
        if ep.actions.ndim != 2 or ep.states.ndim != 2 or not ep.actions.size or ep.actions.shape != ep.states.shape:
            put_text(canvas, "State/action unavailable", (x + 14, y + 58), 0.5, RED, 1)
            return

        idx = int(np.clip(self.frame_index, 0, ep.actions.shape[0] - 1))
        a, s = ep.actions[idx], ep.states[idx]
        err = np.abs(a - s)
        max_err = max(float(np.max(err)), 1.0)

        row_h, start_y = 25, y + 48
        for j in range(len(a)):
            if start_y + j * row_h > y + h - 22:
                break
            yy = start_y + j * row_h
            label_color = CYAN if j == self.selected_joint else MUTED
            put_text(canvas, f"J{j + 1}", (x + 12, yy), 0.43, label_color, 1)
            put_text(canvas, f"A {a[j]:7.2f}", (x + 48, yy), 0.4, ACTION_COLOR, 1)
            put_text(canvas, f"S {s[j]:7.2f}", (x + 135, yy), 0.4, STATE_COLOR, 1)

            bar_x, bar_y = x + 225, yy - 11
            bar_w, bar_h = max(w - 280, 40), 8
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GRID, -1)
            frac = float(err[j] / max_err)
            fill = int(np.clip(frac, 0, 1) * bar_w)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), GREEN if frac < 0.5 else YELLOW, -1)
            put_text(canvas, f"{err[j]:.2f}", (bar_x + bar_w + 8, yy), 0.38, TEXT, 1)

    def draw_validation_panel(self, canvas, x, y, w, h):
        draw_panel(canvas, x, y, w, h, "TECHNICAL VALIDATION")
        ep, m = self.episode, self.episode.metrics

        lines = [
            ("Errors", str(len(ep.errors)), RED if ep.errors else GREEN),
            ("Warnings", str(len(ep.warnings)), YELLOW if ep.warnings else GREEN),
            ("Samples", str(ep.num_samples), TEXT),
            ("Duration", f"{ep.duration:.2f} s", TEXT),
            ("Mean rate", f"{m.get('mean_hz', 0.0):.2f} Hz", GREEN if m.get("mean_hz", 0.0) >= self.args.target_fps * 0.85 else YELLOW),
            ("Max dt", f"{m.get('max_dt_ms', 0.0):.1f} ms", TEXT),
            ("Timing gaps", str(m.get("large_timing_gaps", 0)), GREEN if m.get("large_timing_gaps", 0) == 0 else YELLOW),
            ("Action jumps", str(m.get("flagged_action_jumps", 0)), GREEN if m.get("flagged_action_jumps", 0) == 0 else YELLOW),
            ("Track err mean", f"{m.get('mean_tracking_error', 0.0):.2f}", TEXT),
            ("Track err max", f"{m.get('max_tracking_error', 0.0):.2f}", TEXT),
        ]

        yy = y + 48
        for key, value, color in lines:
            put_text(canvas, key, (x + 14, yy), 0.42, MUTED, 1)
            put_text(canvas, value, (x + w - 145, yy), 0.43, color, 1)
            yy += 23

        issue_y = yy + 3
        all_issues = [("ERR", e, RED) for e in ep.errors] + [("WARN", w_, YELLOW) for w_ in ep.warnings]
        for kind, msg, color in all_issues[:4]:
            put_text(canvas, f"{kind}: {msg[:42]}", (x + 14, issue_y), 0.35, color, 1)
            issue_y += 18

    def draw_timeline(self, canvas, x, y, w, h):
        draw_panel(canvas, x, y, w, h, None)
        n = max(self.episode.num_samples, 1)
        progress = self.frame_index / max(n - 1, 1)
        bar_x, bar_y, bar_w, bar_h = x + 18, y + 17, w - 36, 9
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GRID, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(progress * bar_w), bar_y + bar_h), CYAN, -1)

        t_rel = 0.0
        if len(self.episode.timestamps) > 0 and self.frame_index < len(self.episode.timestamps):
            t_rel = float(self.episode.timestamps[self.frame_index] - self.episode.timestamps[0])

        state = "PLAY" if self.playing else "PAUSE"
        put_text(canvas, f"{state}   frame {self.frame_index + 1}/{n}", (x + 18, y + 48), 0.45, TEXT, 1)
        put_text(canvas, f"{t_rel:.2f}s / {self.episode.duration:.2f}s", (x + w - 180, y + 48), 0.45, MUTED, 1)

    def draw_footer(self, canvas):
        y = self.canvas_h - 42
        cv2.rectangle(canvas, (0, y), (self.canvas_w, self.canvas_h), (15, 17, 20), -1)
        put_text(canvas, "SPACE play/pause   ←/→ frame   J/L ±10   A/D episode   1-9 joint", (18, y + 17), 0.38, MUTED, 1)
        put_text(canvas, "G GOOD   B BAD   C CAMERA   M MOTION   F TASK FAIL   U CLEAR   Q QUIT", (18, y + 34), 0.38, MUTED, 1)

    def render(self):
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        canvas[:] = BG
        self.draw_header(canvas)

        margin = 12
        top = 80
        bottom_footer = 54
        timeline_h = 62
        body_h = self.canvas_h - top - bottom_footer - timeline_h - margin

        left_w = int(self.canvas_w * 0.56)
        right_x = margin + left_w + margin
        right_w = self.canvas_w - right_x - margin

        self.draw_camera_area(canvas, margin, top, left_w, body_h)

        right_top_h = int(body_h * 0.43)
        right_mid_h = int(body_h * 0.31)
        right_bottom_h = body_h - right_top_h - right_mid_h - 2 * margin

        self.draw_joint_plot(canvas, right_x, top, right_w, right_top_h)
        self.draw_tracking_panel(canvas, right_x, top + right_top_h + margin, right_w, right_mid_h)
        self.draw_validation_panel(canvas, right_x, top + right_top_h + margin + right_mid_h + margin, right_w, right_bottom_h)

        timeline_y = top + body_h + margin
        self.draw_timeline(canvas, margin, timeline_y, self.canvas_w - 2 * margin, timeline_h)
        self.draw_footer(canvas)
        return canvas

    def handle_key(self, key):
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            self.playing = not self.playing
            self.last_tick = time.time()
        elif key in (81, 2424832):
            self.step_frame(-1)
        elif key in (83, 2555904):
            self.step_frame(+1)
        elif key == ord("j"):
            self.step_frame(-10)
        elif key == ord("l"):
            self.step_frame(+10)
        elif key == ord("a"):
            self.change_episode(-1)
        elif key == ord("d"):
            self.change_episode(+1)
        elif ord("1") <= key <= ord("9"):
            joint = key - ord("1")
            if joint < self.episode.num_joints:
                self.selected_joint = joint
        elif key == ord("g"):
            self.set_review_tag("GOOD")
        elif key == ord("b"):
            self.set_review_tag("BAD")
        elif key == ord("c"):
            self.set_review_tag("BAD_CAMERA")
        elif key == ord("m"):
            self.set_review_tag("BAD_MOTION")
        elif key == ord("f"):
            self.set_review_tag("BAD_TASK")
        elif key == ord("u"):
            self.review.pop(self.episode.name, None)
            save_review_file(self.review_path, self.review)
        elif key == ord("s"):
            save_review_file(self.review_path, self.review)
        return True

    def run(self):
        try:
            running = True
            while running:
                self.update_playback()
                cv2.imshow(WINDOW_NAME, self.render())
                key = cv2.waitKeyEx(1)
                if key != -1:
                    low = key & 0xFF
                    running = self.handle_key(key if key in (81, 83, 2424832, 2555904) else low)
        finally:
            if self.episode is not None:
                self.episode.close()
            save_review_file(self.review_path, self.review)
            cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(description="Visual validator / inspector for SO-ARM101 teleoperation datasets.")
    parser.add_argument("--dataset-dir", type=str, default="data/pick_place_front_view_v3", help="Dataset folder containing episode_xxxx directories.")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Expected control/data rate used for timing validation.")
    parser.add_argument("--max-action-jump", type=float, default=25.0, help="Flag absolute per-frame joint action changes larger than this value.")
    parser.add_argument("--max-dt-factor", type=float, default=2.5, help="Flag timestamp gaps larger than (1/target_fps) * this factor.")
    parser.add_argument("--width", type=int, default=1440, help="Inspector window render width.")
    parser.add_argument("--height", type=int, default=900, help="Inspector window render height.")
    return parser


def main():
    args = build_parser().parse_args()
    print("=" * 64)
    print("SO-ARM101 DATASET INSPECTOR")
    print("=" * 64)
    print("Dataset:", os.path.abspath(args.dataset_dir))
    print("Episodes:", len(discover_episodes(os.path.abspath(args.dataset_dir))))
    print("\nControls:")
    print("  SPACE      play / pause")
    print("  LEFT/RIGHT previous / next frame")
    print("  J / L      -10 / +10 frames")
    print("  A / D      previous / next episode")
    print("  1..9       select joint plot")
    print("  G          GOOD")
    print("  B          BAD")
    print("  C          BAD_CAMERA")
    print("  M          BAD_MOTION")
    print("  F          BAD_TASK")
    print("  U          clear review")
    print("  Q / ESC    quit\n")

    DatasetInspector(args).run()


if __name__ == "__main__":
    main()