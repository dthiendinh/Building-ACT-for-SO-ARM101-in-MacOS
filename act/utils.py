import numpy as np
import torch
import os
import cv2
from torch.utils.data import DataLoader

import IPython
e = IPython.embed

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
CANONICAL_JOINT_NAMES = [f"{name}.pos" for name in MOTOR_NAMES]
LEGACY_ALPHABETICAL_JOINT_NAMES = sorted(CANONICAL_JOINT_NAMES)


def canonicalize_joint_columns(data):
    """Return states/actions in the canonical physical joint order.

    New recordings persist ``joint_names``. Older recordings from this project
    used sorted(action.keys()), so files without metadata use that known legacy
    order for backward compatibility.
    """
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    if states.ndim != 2 or states.shape[1] != len(CANONICAL_JOINT_NAMES):
        raise ValueError(f"Expected states with shape (T, 6), got {states.shape}")
    if actions.shape != states.shape:
        raise ValueError(f"Action/state shape mismatch: {actions.shape} vs {states.shape}")
    source_names = (
        [str(name) for name in data["joint_names"].tolist()]
        if "joint_names" in data.files
        else LEGACY_ALPHABETICAL_JOINT_NAMES
    )

    if len(source_names) != len(CANONICAL_JOINT_NAMES) or set(source_names) != set(CANONICAL_JOINT_NAMES):
        raise ValueError(f"Unsupported joint schema in dataset: {source_names}")

    indices = [source_names.index(name) for name in CANONICAL_JOINT_NAMES]
    return states[:, indices], actions[:, indices]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size):
        super().__init__()
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = int(chunk_size)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        """
        Return one training sample for ACT.

        Output:
            image_data: (num_cameras, 3, H, W)
            qpos_data: (state_dim, )
            action_data: (chunk_size, action_dim)
            is_pad: (chunk_size,)
        """
        # Determine which episode and timestep this index uses
        episode_id = self.episode_ids[index]
        
        episode_dir = os.path.join(self.dataset_dir, f"episode_{episode_id:04d}")

        data_path = os.path.join(episode_dir, "data.npz")

        with np.load(data_path) as data:
            states, actions = canonicalize_joint_columns(data)
        episode_len = len(actions)
        if episode_len == 0:
            raise ValueError(f"Episode has no samples: {data_path}")

        # Recorded episode length varies; only the model's action chunk is fixed.
        start_ts = np.random.choice(episode_len)

        qpos = states[start_ts]  # Shape (6,)
        action = actions[start_ts:start_ts + self.chunk_size]
        action_len = len(action)
        padded_action = np.zeros((self.chunk_size, actions.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(self.chunk_size, dtype=bool)
        is_pad[:action_len] = False


        # Read camera
        all_cam_images = []

        for cam_name in self.camera_names:
            video_path = os.path.join(episode_dir, f"{cam_name}.mp4")
            cap = cv2.VideoCapture(video_path)

            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open video: {video_path}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_ts)
                ret, frame = cap.read()
            finally:
                cap.release()
            if not ret:
                raise RuntimeError(
                    f"Cannot read frame {start_ts} "
                    f"from {video_path}"
                )
            # OpenCV
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_cam_images.append(frame)
        # Before transpose: (num_cameras, H, W, 3)
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last 
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # Normalize image and change dtype to float
        image_data = image_data / 255.0
        action_mean = torch.as_tensor(self.norm_stats["action_mean"], dtype=torch.float32)
        action_std = torch.as_tensor(self.norm_stats["action_std"], dtype=torch.float32)
        qpos_mean = torch.as_tensor(self.norm_stats["qpos_mean"], dtype=torch.float32)
        qpos_std = torch.as_tensor(self.norm_stats["qpos_std"], dtype=torch.float32)
        action_data = (action_data - action_mean) / action_std
        action_data[is_pad] = 0.0
        qpos_data = (qpos_data - qpos_mean) / qpos_std

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, episode_ids=None):
    """Compute per-joint statistics over real timesteps, without episode padding.

    episode_ids lets the loader use only the training split for normalization.
    """
    all_qpos_data = []
    all_action_data = []
    if episode_ids is None:
        episode_ids = range(num_episodes)
    for episode_idx in episode_ids:
        episode_dir = os.path.join(dataset_dir, f"episode_{episode_idx:04d}")
        data_path = os.path.join(episode_dir, "data.npz")
        with np.load(data_path) as data:
            qpos, actions = canonicalize_joint_columns(data)
        if len(actions) == 0:
            raise ValueError(f"Episode has no samples: {data_path}")
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(actions))
    if not all_qpos_data:
        raise ValueError("At least one episode is required for normalization statistics")
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0)
    action_std = all_action_data.std(dim=0, correction=0)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0)
    qpos_std = all_qpos_data.std(dim=0, correction=0)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
            "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos,
             "joint_names": CANONICAL_JOINT_NAMES}
            
    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size):
    print(f'\nData from: {dataset_dir}\n')
    if num_episodes < 2:
        raise ValueError("At least two episodes are required for separate train/validation splits")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action 
    norm_stats = get_norm_stats(dataset_dir, num_episodes, episode_ids=train_indices)

    # construct dataset and dataloader 
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers = 1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers = 1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats


def aggregate_temporal_actions(actions_for_curr_step, decay=0.01):
    """Average chronological predictions using the actions' dtype and device.

    The caller selects predictions by query time, so zero-valued actions remain
    valid. Creating weights in PyTorch avoids NumPy float64 tensors on MPS.
    """
    if actions_for_curr_step.ndim != 2 or len(actions_for_curr_step) == 0:
        raise ValueError("Temporal aggregation requires at least one action vector")
    indices = torch.arange(
        len(actions_for_curr_step),
        device=actions_for_curr_step.device,
        dtype=actions_for_curr_step.dtype,
    )
    weights = torch.softmax(-decay * indices, dim=0).unsqueeze(1)
    return (actions_for_curr_step * weights).sum(dim=0, keepdim=True)



### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
 
