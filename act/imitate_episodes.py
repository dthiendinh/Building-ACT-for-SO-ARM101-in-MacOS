# Allow direct execution from the repository as well as python -m.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np 
import os
import pickle
import json
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from act.utils import load_data  # data functions
from act.utils import compute_dict_mean, set_seed, detach_dict, aggregate_temporal_actions  # helper functions
from act.policy import ACTPolicy, CNNMLPPolicy

import IPython
e = IPython.embed

# Constant
from hardware_constant import CONTROL_DT, CAMERA_CONFIGS
DT = CONTROL_DT

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = camera_names[0]

    # load trained policy 
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(loading_status)
    policy.to(device)
    policy.eval()
    print(f"Loaded: {ckpt_path}")

    # Load dataset's mean/std
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    from act.utils import CANONICAL_JOINT_NAMES
    if stats.get('joint_names') != CANONICAL_JOINT_NAMES:
        raise ValueError(
            "dataset_stats.pkl has no verified canonical SO-ARM101 joint schema. "
            "Retrain/regenerate the stats with the updated loader; refusing unsafe inference."
        )

    # normalize qpos
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

    # denormalize qpos
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    from act.low_level_control import make_real_env
    from act.low_level_control import (
        ROBOT_PORT,
        ROBOT_ID,
        CAMERAS,
        RESET_POSE,
        MAX_RELATIVE_TARGET,
        DT,
    )
    reset_pose = config.get('reset_pose')
    if reset_pose is None:
        reset_pose = RESET_POSE
    if reset_pose is None:
        raise ValueError(
            "Evaluation on the real robot requires --reset_pose with six values."
        )
    max_relative_target = config.get('max_relative_target', MAX_RELATIVE_TARGET)
    num_rollouts = config.get('num_rollouts', 1)
    if num_rollouts < 1:
        raise ValueError("num_rollouts must be at least 1")
    if max_relative_target is None or not np.isfinite(max_relative_target) or max_relative_target <= 0:
        raise ValueError("max_relative_target must be a finite positive number")
    if np.asarray(reset_pose).shape != (6,) or not np.isfinite(reset_pose).all():
        raise ValueError("reset_pose must contain six finite values")
    env = make_real_env(
        port=ROBOT_PORT,
        robot_id=ROBOT_ID,
        cameras=CAMERAS,
        reset_pose=reset_pose,
        dt=DT,
        max_relative_target=max_relative_target,
    )
    
    # query every 100 steps
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        # if temporal aggregation, query every each step
        query_frequency = 1
        num_queries = policy_config['num_queries']


    max_timesteps = int(max_timesteps * 1) # may increase for real_world tasks

    """
    Every rollout: reset -> run an episode -> cal reward -> save result
    """
    episode_returns = []
    highest_rewards = []
    try:
      for rollout_id in range(num_rollouts):
        ts = env.reset()

        plt_img = None
        if onscreen_render:
            _, ax = plt.subplots()
            plt_img = ax.imshow(ts.observation["images"][onscreen_cam])
            plt.ion()
        
        # evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).to(device)

        qpos_history = torch.zeros([1, max_timesteps, state_dim]).to(device)
        image_list = []    # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                # update onscreen render and wait for DT
                if onscreen_render:
                    image = (ts.observation["images"][onscreen_cam])
                    plt_img.set_data(image)
                    plt.pause(0.001)

                # process previous timestep to get qpos and image_list
                # ts includes qpos + images
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})

                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                # query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        # infer every frequency
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        # Temporal Aggregation
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        # Every query predicts num_queries steps. Select only
                        # queries covering t, including legitimate zero actions.
                        first_query = max(0, t - num_queries + 1)
                        actions_for_curr_step = all_time_actions[first_query:t + 1, t]
                        raw_action = aggregate_temporal_actions(actions_for_curr_step)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError
                
                # post-process actions
                # (1, action_dim) -> (action_dim)
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                # Step the environment
                ts = env.step(target_qpos)

                # for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                # just for evaluating how well the model is 
                rewards.append(ts.reward)
            
            if onscreen_render:
                plt.close()
            
        measured_rewards = [reward for reward in rewards if reward is not None]
        if measured_rewards:
            episode_return = float(np.sum(measured_rewards))
            episode_highest_reward = float(np.max(measured_rewards))
            episode_returns.append(episode_return)
            highest_rewards.append(episode_highest_reward)
            print(f'Rollout {rollout_id}: return={episode_return}, highest_reward={episode_highest_reward}')
        else:
            print(f'Rollout {rollout_id}: completed; reward is not configured')
        
        # TODO define this func
        if save_episode and 'save_videos' in globals():
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
        elif save_episode:
            print("Video was not saved because save_videos() is not configured.")
    finally:
        env.close()

    success_rate = None
    avg_return = float(np.mean(episode_returns)) if episode_returns else None
    summary_str = f'\nCompleted rollouts: {num_rollouts}\n'
    if avg_return is None:
        summary_str += 'Reward/success metrics unavailable: no real-world reward function is configured.\n'
    else:
        summary_str += f'Average return: {avg_return}\n'
    
    print(summary_str)

    # save success rate to txt 
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))
    
    return success_rate, avg_return

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    with tqdm(total=num_epochs, desc="Training", dynamic_ncols=True) as progress:
        for epoch in range(num_epochs):

            # Training
            policy.train()
            optimizer.zero_grad()
            for batch_idx, data in enumerate(train_dataloader):
                forward_dict = forward_pass(data, policy)

                # Backward
                loss = forward_dict['loss']
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss; check dataset and hyperparameters')
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_history.append(detach_dict(forward_dict))
            epoch_summary = compute_dict_mean(train_history[(batch_idx+1) * epoch: (batch_idx+1)*(epoch+1)])
            epoch_train_loss = epoch_summary['loss']


            #validation
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

            if epoch % 100 == 0:
                ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
                torch.save(policy.state_dict(), ckpt_path)
                plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

            progress.set_postfix(
                train=f"{epoch_train_loss.item():.4f}",
                val=f"{epoch_val_loss.item():.4f}",
                best=f"{float(min_val_loss):.4f}",
                refresh=False,
            )
            progress.update(1)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    if best_ckpt_info is None:
        raise RuntimeError("No finite validation loss; check dataset and training settings")
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # Save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()





def main(args):
    set_seed(args["seed"])

    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    from act.low_level_control import TASK_CONFIGS
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    state_dim = task_config["state_dim"]

    # fixed parameters
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if state_dim != 6:
        raise ValueError('SO-ARM101 requires state_dim=6')
    if not camera_names or not set(camera_names).issubset(CAMERA_CONFIGS):
        raise ValueError('Task cameras must be present in hardware_constant.CAMERA_CONFIGS')
    if num_epochs < 1 or batch_size_train < 1 or not np.isfinite(args['lr']) or args['lr'] <= 0:
        raise ValueError("num_epochs, batch_size and lr must be positive")
    if policy_class == 'ACT':
        for key in ('chunk_size', 'hidden_dim', 'dim_feedforward'):
            if args.get(key) is None or args[key] < 1:
                raise ValueError(f"--{key} must be a positive integer for ACT")
        if args.get('kl_weight') is None or args['kl_weight'] < 0:
            raise ValueError("--kl_weight must be non-negative for ACT")
        if args['hidden_dim'] % 8:
            raise ValueError("--hidden_dim must be divisible by 8")
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        'state_dim': state_dim,
                        }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 
                        'lr_backbone': lr_backbone, 
                        'backbone': backbone, 
                        'num_queries': 1,
                        'camera_names' : camera_names}
    else:
        raise NotImplementedError
    
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'num_rollouts': args['num_rollouts'],
        'reset_pose': args['reset_pose'],
        'max_relative_target': args['max_relative_target'],
        'dataset_dir': dataset_dir,
        'num_episodes': num_episodes,
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])
        
        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,
        chunk_size=policy_config['num_queries'],
    )

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=1)
    parser.add_argument('--reset_pose', type=float, nargs=6)
    parser.add_argument('--max_relative_target', type=float, default=5.0)
    
    main(vars(parser.parse_args()))
