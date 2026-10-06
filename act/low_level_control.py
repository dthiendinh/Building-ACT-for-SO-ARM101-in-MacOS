from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
import numpy as np
import time
import dm_env


# FOR SO-ARM101
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# CONSTANT for CONFIG
ROBOT_PORT = "/dev/tty.usbmodem5B8E1131141"
ROBOT_ID = "so101_follower"
RESET_POSE = None
DT = 1/30 # 30Hz
MAX_RELATIVE_TARGET = 5.0  # degrees per command; clipped again by LeRobot
CAMERA_FPS = 30
CAMERAS = {
    "front": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=CAMERA_FPS),
    "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=CAMERA_FPS),
}
TASK_CONFIGS = {

    "so101_pick_place": {

        "dataset_dir":
            "teleoperation/data/pick_place_front_view_v3",

        "num_episodes":
            50,

        # Evaluation rollout horizon in control steps (~10 s at 30 Hz).
        # Recorded episodes have their own lengths; training uses chunk_size.
        "episode_len":
            300,

        "camera_names":
            ["wrist", "front"],

        "state_dim":
            6,
    },
}


class RealEnv:
    # Create the real robot
    def __init__(self, port, robot_id, cameras=None, dt=1/30, reset_pose=None, max_relative_target=None):
        self.dt = dt
        self.reset_pose = reset_pose

        # SO-ARM configuration
        config = SO101FollowerConfig(
            port = port,
            id = robot_id,
            cameras = cameras,
            max_relative_target = max_relative_target,
        )
        
        self.robot = SO101Follower(config)
        self.robot.connect()

    
    def get_qpos(self, raw_obs):
        qpos = np.array(
            [
                raw_obs[f"{name}.pos"]
                for name in MOTOR_NAMES
            ],
            dtype=np.float32
        )

        return qpos

    def get_images(self, raw_obs):

        images = {}

        for camera_name in self.robot.cameras:
            images[camera_name] = raw_obs[camera_name]

        return images
    
    def get_observation(self):
        """
        Read SO-ARM observation exactly once and convert it into format expected by ACT
        """

        raw_obs = self.robot.get_observation()

        obs = {
            "qpos": self.get_qpos(raw_obs),
            "images": self.get_images(raw_obs),
        }

        return obs

    # Action
    def _vector_to_action_dict(self, action):
        """
        Convert np.ndarray(6,) into 
        { 
            "shoulder_pan.pos": ...,
            ...
            "gripper.pos":....
        }
        """
        action = np.asarray(
            action,
            dtype = np.float32
        )

        if action.shape != (6,):
            raise ValueError(
                "SO-ARM101 action must have "
                f"shape (6,), got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("SO-ARM101 action contains NaN or infinity")
        
        robot_action = {
            f"{name}.pos": float(action[i])
            for i, name in enumerate(MOTOR_NAMES)
        }

        return robot_action

    # Reset
    def _move_to_pose(self, target_pos, move_time=1.5):
        "Smoothly interpolate from current position to target position."

        target_pose = np.asarray(target_pos,dtype=np.float32)
        if target_pose.shape != (6,):
            raise ValueError(
                "reset pose must have shape (6,)"
            )
        
        raw_obs = self.robot.get_observation()

        start_pos = self.get_qpos(raw_obs)

        num_steps = max(1, int(move_time/self.dt))

        for i in range(1, num_steps + 1):
            alpha = i / num_steps

            pose = (1.0 - alpha) * start_pos + alpha * target_pos

            command = self._vector_to_action_dict(pose)

            self.robot.send_action(command)

            time.sleep(self.dt)
        
    def _reset_joints(self):
        # Move robot to configured reset pose
        if self.reset_pose is None:
            raise RuntimeError(
                "A six-joint reset_pose is required before a real-robot rollout. "
                "Refusing to start from an unknown pose."
            )
        
        self._move_to_pose(self.reset_pose)

    # Reward to evaluate
    def get_reward(self):
        # No automatic real-world task reward yet.
        # None explicitly means "not measured"; returning 0 would make every
        # rollout look successful when env_max_reward is also zero.
        return None
    
    def reset(self, fake=False):
        """
        ACT calls this at the beginning
        of an episode.

        fake=True:
            do not physically move robot.

        fake=False:
            move robot to reset_pose if provided.
        """

        if not fake:
            self._reset_joints()

        # container TimeStep
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )
    
    def step(self, action):
        # send one 6D ACT action to SO-ARM101
        robot_action = self._vector_to_action_dict(action)
        self.robot.send_action(robot_action)
        time.sleep(self.dt)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def close(self):
        if self.robot.is_connected:
            self.robot.disconnect()


def make_real_env(port, robot_id, cameras=None, dt=1 / 30, reset_pose=None, max_relative_target=None):
    env = RealEnv(
        port=port,
        robot_id=robot_id,
        cameras=cameras,
        dt=dt,
        reset_pose=reset_pose,
        max_relative_target=max_relative_target,
    )

    return env
