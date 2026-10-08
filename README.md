# ACT for SO-ARM101 on macOS

Collect demonstrations with an SO-ARM101 leader/follower pair, review them locally, and train an **ACT (Action Chunking with Transformers)** policy on a Mac. The policy observes the follower's six joint positions and two RGB cameras (`wrist` and `front`), then predicts a chunk of future joint targets.

LeRobot provides the motor and camera drivers. This repository provides its own recorder, NPZ/MP4 dataset format, inspector, ACT model, training loop, and real robot environment. No simulator is required. These datasets are not directly compatible with `lerobot-train` or `LeRobotDataset`.

## Workflow

1. Install the Python environment.
2. Find USB ports and camera indices; update the shared hardware constants.
3. Set up motors if necessary, then calibrate both arms.
4. Check teleoperation, record demonstrations, and review every episode.
5. Copy approved demonstrations into a training dataset and configure the task.
6. Run a one-epoch smoke test, then train a policy.
7. Evaluate the trained policy on the follower using a verified reset pose.

## Requirements and verification status

- macOS; an Apple Silicon Mac with MPS support is the intended training platform. CPU is the fallback.
- Python **3.12**. The dependency set in this project was checked with Python 3.12.14; use that Python series rather than 3.14.
- An assembled SO-ARM101 leader/follower pair, USB connections, and the correct power supplies for the installed motors.
- Two cameras accessible through OpenCV, mounted consistently at the wrist and in front of the workspace.
- Camera permission for the terminal or application running Python, under **System Settings → Privacy & Security → Camera**.

## 1. Install the environment

Install Conda or Miniforge first, then run:

```bash
git clone https://github.com/dthiendinh/Building-ACT-for-SO-ARM101-in-MacOS.git
cd Building-ACT-for-SO-ARM101-in-MacOS

conda create -n virtual_environment python=3.12 -y
conda activate virtual_environment
python -m pip install -r requirements.txt
python -m pip check
```

If the repository is already cloned, change into that checkout instead. Run the remaining commands from the **repository root**, with the environment activated. The `python -m ...` commands below and direct script execution are both supported.

Check imports and the training device without connecting a robot:

```bash
python -c "import torch, torchvision, cv2, lerobot; print('PyTorch:', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
python -m teleoperation.record_teleop_data --help
python -m teleoperation.data_inspector --help
python -m act.imitate_episodes --help
```

On first use, training may download pretrained ResNet18 weights to the PyTorch cache. The code selects MPS automatically when available, otherwise CPU; there is currently no `--device` option or CUDA selection.

## 2. Set the hardware constants

Edit **[hardware_constant.py](hardware_constant.py)**. It is the shared configuration used by teleoperation, recording, the loader's joint schema, and robot evaluation.

| Constant | Purpose | Repository default |
| --- | --- | --- |
| `LEADER_PORT` | Leader USB serial port | `/dev/tty.usbmodem5B8E1128291` |
| `FOLLOWER_PORT` | Follower USB serial port | `/dev/tty.usbmodem5B8E1131141` |
| `LEADER_ID` | Calibration identity for the leader | `so101_leader` |
| `FOLLOWER_ID` | Calibration identity for the follower | `so101_follower` |
| `CAMERA_CONFIGS` | Camera names, indices, width, height, FPS | `wrist`: 0; `front`: 1; both 640×480 |
| `CAMERA_FPS` | Encoded video FPS and requested camera FPS | 30 |
| `CONTROL_HZ` / `CONTROL_DT` | Target control frequency / interval | 30 Hz / `1 / CONTROL_HZ` |
| `MOTOR_NAMES` / `JOINT_KEYS` | Physical ordering of state and action columns | Six joints, listed below |

The serial ports and camera indices above are machine-specific examples. Discover your actual ports:

```bash
python -m serial.tools.list_ports -v
lerobot-find-port
```

Use `lerobot-find-port` once for each arm and follow its unplug/replug instructions. Update the ports in `hardware_constant.py`; the discovery command does not update the file for you.

Discover cameras, then preview each index separately:

```bash
lerobot-find-cameras opencv
python -m testing.camtest --index 0 --width 640 --height 480 --fps 30
python -m testing.camtest --index 1 --width 640 --height 480 --fps 30
```

Press `q` to close a preview before opening the next one. Confirm which image is the wrist view and which is the front view, then assign those indices in `CAMERA_CONFIGS`. Recheck after reconnecting cameras. Both streams must use the **same image dimensions**, including after any configured rotation, because the recorder and loader stack them together. Keep camera color mode RGB: the recorder converts to BGR for OpenCV video encoding, and the loader converts back to RGB.

For the commands in the next section, load your edited values into the current shell:

```bash
export LEADER_PORT="$(python -c 'from hardware_constant import LEADER_PORT; print(LEADER_PORT)')"
export FOLLOWER_PORT="$(python -c 'from hardware_constant import FOLLOWER_PORT; print(FOLLOWER_PORT)')"
export LEADER_ID="$(python -c 'from hardware_constant import LEADER_ID; print(LEADER_ID)')"
export FOLLOWER_ID="$(python -c 'from hardware_constant import FOLLOWER_ID; print(FOLLOWER_ID)')"
```

These variables are only a convenience for the LeRobot CLI commands. The project scripts read `hardware_constant.py` directly.

## 3. Set up and calibrate the arms

### Motor setup: new or reconfigured motors only

If motor IDs and baud rates have not been configured, follow the [official LeRobot SO-101 setup guide](https://huggingface.co/docs/lerobot/so101), including its motor connection sequence. Run these commands separately for the appropriate arm:

```bash
lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT"

lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT"
```

Follow the interactive instructions to assign each motor. This is a hardware setup operation; it is unnecessary for an arm whose motors are already configured correctly.

### Calibration

Place the arms where you can move their joints freely and follow the calibration prompts. Start with the follower, then the leader:

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID"

lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID"
```

When prompted, place the joints around the middle of their travel, confirm, and move them through the ranges requested by the CLI. Use the [official calibration walkthrough](https://huggingface.co/docs/lerobot/so101) for the physical poses and video reference. If calibration already exists, the CLI offers reuse or recalibration; choose according to whether that file belongs to the same arm and assembly.

For the pinned LeRobot 0.6.1 driver, the default files are:

```text
~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<LEADER_ID>.json
~/.cache/huggingface/lerobot/calibration/robots/so_follower/<FOLLOWER_ID>.json
```

A custom Hugging Face/LeRobot cache location can change that base path. Keep the same IDs in calibration and `hardware_constant.py`; changing an ID selects a different calibration file. The project scripts use the driver's default calibration directory.

## 4. Check teleoperation

```bash
python -B -m teleoperation.teleop
```

The follower starts tracking the leader after connection. Move the leader slowly and verify that corresponding joints and the gripper track correctly. Stop with `Ctrl+C`. Close this process before starting recording or evaluation: one process should own the follower port at a time.

The driver uses degrees for the five arm joints and a normalized **0–100** range for the gripper. The canonical vector order is:

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

Keep this order consistent between recording, training, reset poses, and inference. New recordings persist it as `joint_names` metadata.

## 5. Record demonstrations

```bash
python -B -m teleoperation.record_teleop_data --data-dir pick_place_act_v1
```

`--data-dir` names the dataset folder under `teleoperation/data/`. The recorder connects both arms and both cameras, opens a dashboard, and starts teleoperation immediately, including when no episode is being recorded.

Click/focus the OpenCV window to use these keys:

| Key | Action |
| --- | --- |
| `r` | Start a new episode |
| `s` | Stop and save the current episode |
| `x` | Discard the current episode |
| `q` | Quit and save an active episode |
| `Ctrl+C` in the terminal | Stop; attempt to save an active episode and disconnect |

Record one complete successful task per episode, for example approach → grasp → lift → transfer → release. Start recording before the motion and stop after completion. Reset the objects and arm between episodes. Keep the lighting, camera mounting, and workspace consistent, while varying object positions within the task's intended range.

The dashboard displays loop frequency, sample count, tracking, and writer queue size. Video encoding runs on a background thread. A full queue stops collection with an error instead of silently dropping frames; inspect that episode before using it. Empty episodes are discarded. Existing episode directories are preserved, and new recordings continue with the next index.

Output:

```text
teleoperation/data/pick_place_act_v1/
├── episode_0000/
│   ├── data.npz
│   ├── wrist.mp4
│   └── front.mp4
├── episode_0001/
│   └── ...
└── dataset_review.json       # Created during review
```

Each `data.npz` contains:

| Key | Shape | Contents |
| --- | --- | --- |
| `timestamps` | `(T,)` | Sample timestamps, in seconds |
| `states` | `(T, 6)` | Measured follower joint positions before the command |
| `actions` | `(T, 6)` | Leader joint targets sent to the follower |
| `joint_names` | `(6,)` | Column order |

Each MP4 has one frame per sample. The video is encoded at `CAMERA_FPS`; actual control timing is recorded separately in `timestamps`. Large timing gaps still matter even when video frame counts match. `teleoperation/follower_state.json` is also updated with the latest measured joints for local inspection.

## 6. Review and prepare a training dataset

```bash
python -m teleoperation.data_inspector \
  --dataset-dir teleoperation/data/pick_place_act_v1
```

| Key | Action |
| --- | --- |
| `Space` | Play/pause |
| Left / Right; `j` / `l` | Step one frame; jump ten frames |
| `a` / `d` | Previous/next episode |
| `1`–`9` | Select a joint plot |
| `g` | Mark GOOD |
| `b` / `c` / `m` / `f` | Mark BAD / BAD_CAMERA / BAD_MOTION / BAD_TASK |
| `u` | Clear the review tag |
| `s` | Save reviews |
| `q` / Esc | Save and quit |

Inspect both camera streams, joint traces, timing, and whether the task actually succeeded. The inspector's technical status is not a task-success label. A camera can be optional to the inspector but required by your ACT task.

**The trainer does not automatically filter `dataset_review.json`.** Create a separate dataset containing only GOOD episodes, numbered consecutively from `episode_0000`. This example copies approved episodes and preserves the source:

```bash
python - <<'PY'
import json
import shutil
from pathlib import Path

source = Path("teleoperation/data/pick_place_act_v1")
output = Path("teleoperation/data/pick_place_act_v1_train")
review = json.loads((source / "dataset_review.json").read_text())
approved = []
for episode in sorted(source.glob("episode_*")):
    tag = review.get(episode.name, {})
    tag = tag.get("tag") if isinstance(tag, dict) else tag
    if episode.is_dir() and tag == "GOOD":
        approved.append(episode)
if len(approved) < 2:
    raise SystemExit("Review at least two usable episodes as GOOD first.")
output.mkdir(parents=True, exist_ok=False)
for index, episode in enumerate(approved):
    shutil.copytree(episode, output / f"episode_{index:04d}")
print(f"Created {output}; set num_episodes = {len(approved)}")
PY
```

Use a new output directory when rebuilding the selection. At least two episodes are required for separate train/validation splits; this minimum only allows the code to run. A useful policy needs many varied, successful demonstrations and assessment on held-out task setups.

## 7. Configure the training task

Edit `TASK_CONFIGS["so101_pick_place"]` in **[act/low_level_control.py](act/low_level_control.py)**:

```python
"so101_pick_place": {
    "dataset_dir": "teleoperation/data/pick_place_act_v1_train",
    "num_episodes": 50,                # Replace with the count printed above
    "episode_len": 300,                # Evaluation steps, not training truncation
    "camera_names": ["wrist", "front"],
    "state_dim": 6,
},
```

`num_episodes=50` is a configuration example, not an ACT requirement. Set it to the actual approved count. The loader expects exactly `episode_0000` through `episode_{N-1:04d}` to exist. Paths are relative to the current working directory, so run training from the repository root.

Other configuration locations:

| Setting | Location |
| --- | --- |
| Ports, calibration IDs, camera indices/resolution/FPS, control interval | `hardware_constant.py` |
| Dataset path/count, model camera order, evaluation horizon | `TASK_CONFIGS` in `act/low_level_control.py` |
| Default evaluation reset pose | `RESET_POSE` in `act/low_level_control.py`, initially `None`; override with `--reset_pose` |
| Evaluation target limit | `--max_relative_target`, default 5.0 |
| Learning rate, epochs, batch size, chunk size, latent loss weight, model width | Trainer CLI, shown below |
| Backbone and Transformer depth | `main()` in `act/imitate_episodes.py`: ResNet18, 4 encoder layers, 7 decoder layers, 8 heads |

Training preflight checks every selected episode for arrays, joint schema, finite values, timestamps, required camera files, frame counts, and consistent image sizes. It does not fully decode every video in advance; frame decoding errors can still appear during loading.

The loader shuffles episodes using the requested seed, splits approximately 80%/20%, and computes normalization statistics **only from training episodes**. Each dataset access samples one random timestep and its following action chunk. Short chunks are padded and masked out of the reconstruction loss. An epoch visits the training episodes once; it is not a sweep over every recorded frame.

Older recordings without `joint_names` are assumed to use this project's legacy alphabetical joint order, then reordered by the loader. Verify that assumption before using recordings from another source.

## 8. Train ACT

First run one epoch to check the complete training path:

```bash
MPLBACKEND=Agg python -B -m act.imitate_episodes \
  --task_name so101_pick_place \
  --ckpt_dir outputs/act_pick_place_smoke \
  --policy_class ACT \
  --batch_size 1 \
  --seed 0 \
  --num_epochs 1 \
  --lr 1e-5 \
  --kl_weight 10 \
  --chunk_size 20 \
  --hidden_dim 128 \
  --dim_feedforward 512
```

This command requires the dataset and task configuration from steps 6–7. Then train into a **new output directory**, for example:

```bash
MPLBACKEND=Agg python -B -m act.imitate_episodes \
  --task_name so101_pick_place \
  --ckpt_dir outputs/act_pick_place_v1 \
  --policy_class ACT \
  --batch_size 1 \
  --seed 0 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --kl_weight 10 \
  --chunk_size 20 \
  --hidden_dim 128 \
  --dim_feedforward 512
```

These are starting settings, not validated task-performance hyperparameters. Adjust the training duration and model settings based on validation loss and physical task trials. `hidden_dim` must be positive and divisible by 8. Reduce batch size, chunk size, or model width if memory is insufficient. `MPLBACKEND=Agg` saves training plots without opening windows.

Training writes:

```text
outputs/act_pick_place_v1/
├── training_config.json       # Task, architecture and run settings
├── dataset_stats.pkl          # Train-only normalization and joint schema
├── policy_best.ckpt           # Lowest validation loss after a training epoch
├── policy_last.ckpt
├── policy_epoch_*_seed_*.ckpt
└── train_val_*_seed_*.png
```

Keep the configuration and statistics with the checkpoints. There is no resume-training CLI; rerunning in the same directory starts a new model and can overwrite outputs. Validation samples random timesteps and a CVAE latent, so its loss can fluctuate. Low loss alone does not establish manipulation success.

A `CNNMLP` baseline is also implemented (`--policy_class CNNMLP`), but ACT is the main workflow documented here.

## 9. Evaluate on the real follower

Evaluation commands physically move the follower. Stop teleoperation/recording first, confirm follower port/calibration and both camera views, clear the workspace, and use a reset pose you have checked on your own assembly.

You can inspect `teleoperation/follower_state.json` after placing the follower in the intended starting pose during teleoperation. Read the named joint values in canonical order; verify that this pose and the route back to it are suitable for your workspace. The environment interpolates from its current pose to the reset pose before each rollout.

In the command below, **replace `PAN LIFT ELBOW WRIST_FLEX WRIST_ROLL GRIPPER` with six numeric values**: degrees for the first five, 0–100 for the gripper. Keep architecture and camera ordering identical to training, as recorded in `training_config.json`.

```bash
python -m act.imitate_episodes \
  --eval \
  --task_name so101_pick_place \
  --ckpt_dir outputs/act_pick_place_v1 \
  --policy_class ACT \
  --batch_size 1 \
  --seed 0 \
  --num_epochs 1 \
  --lr 1e-5 \
  --kl_weight 10 \
  --chunk_size 20 \
  --hidden_dim 128 \
  --dim_feedforward 512 \
  --num_rollouts 1 \
  --max_relative_target 5.0 \
  --reset_pose PAN LIFT ELBOW WRIST_FLEX WRIST_ROLL GRIPPER
```

The shared CLI still requires `batch_size`, `num_epochs`, and `lr` in evaluation mode; it does not train in this mode. It loads `policy_best.ckpt` and `dataset_stats.pkl`. The original dataset files are unnecessary for evaluation. A reset pose is required unless you set `RESET_POSE` in code.

By default, ACT queries once per chunk and executes that chunk sequentially. Add `--temporal_agg` to query at every step and combine overlapping predictions. Add `--onscreen_render` for a live camera view. `episode_len=300` sets 300 control steps; at the requested 30 Hz this is nominally 10 seconds, but inference and I/O add time.

`max_relative_target` limits the target change from the measured position per motor command, including the gripper's own units. It does not check collisions or define a safe workspace. `Ctrl+C` exits through the cleanup path.

The real environment has **no automatic task reward or success detector**. Evaluation reports completed rollouts and writes `result_policy_best.txt`; assess task success yourself. Rollout video saving is not implemented.

## Troubleshooting

| Symptom | Check or fix |
| --- | --- |
| `ModuleNotFoundError` | Activate the Python 3.12 environment, install requirements, and run from the repo root using the documented commands. |
| No robot serial port | Check arm USB cables, power, macOS USB access, and `lerobot-find-port`; update the constants. |
| Wrong calibration or an unexpected calibration prompt | Match the physical arm and its ID to the saved calibration; recalibrate if the assembly changed. |
| Camera cannot open / wrong view | Grant camera permission, close other camera users, recheck indices and supported resolution/FPS. |
| `episode_XXXX/data.npz` missing | Correct `dataset_dir` and `num_episodes`; ensure approved episodes are consecutively numbered. |
| Missing `front.mp4` / video count mismatch | Exclude or re-record incomplete episodes; every camera in `camera_names` must be present for every sample. |
| Low loop frequency / writer queue fills | Check camera throughput, resolution, disk space, and other workload; review timestamps before training. |
| Reset-pose error | Supply six finite, verified values in canonical joint order. |
| Checkpoint size mismatch | Match model width, feedforward size, chunk size, policy type, and camera order to the training configuration. |
| Evaluation rejects joint statistics | Regenerate statistics and retrain with the current canonical joint schema. |

## Repository layout

```text
.
├── hardware_constant.py           # Shared hardware settings and joint schema
├── hardware_utils.py              # Cleanup for full/partial connections
├── requirements.txt
├── teleoperation/
│   ├── teleop.py                   # Leader → follower
│   ├── record_teleop_data.py        # Dashboard and NPZ/MP4 recorder
│   └── data_inspector.py           # Dataset playback and review tags
├── act/
│   ├── imitate_episodes.py         # Training and real-robot evaluation CLI
│   ├── low_level_control.py        # Task configuration and RealEnv
│   ├── policy.py                   # ACT and CNNMLP losses/inference
│   ├── main.py                     # Model and optimizer builders
│   ├── utils.py                    # Loader, validation, statistics, aggregation
│   └── models/                    # ResNet, CVAE, Transformer, position encoding
└── testing/
    └── camtest.py                  # Manual camera preview
```

Generated datasets, outputs, Python caches, and follower snapshots are excluded by `.gitignore`.

## References

- [LeRobot SO-101 setup and calibration](https://huggingface.co/docs/lerobot/so101)
- [LeRobot source](https://github.com/huggingface/lerobot)
- [ACT project and paper](https://tonyzhaozh.github.io/aloha/)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
