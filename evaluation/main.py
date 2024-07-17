from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
import torch
import argparse
from time import sleep
import os
import sys

# Ajouter le chemin du dossier parent
sys.path.append('..')

# Environments
from envs.ur3e_env_reward0_01 import Ur3eEnvRew0_01
from envs.ur3e_env_reward0_05 import Ur3eEnvRew0_05
from envs.ur3e_target_orientation import Ur3eTargetOrientation
from envs.ur3e_curriculum import Ur3eCurriculum

def get_env(env_name, urdf_ur3e_path):
    if env_name == "Ur3eEnvRew0_01":
        return Ur3eEnvRew0_01(urdf_ur3e_path, "UR3e", [0, 0, 0])
    elif env_name == "Ur3eEnvRew0_05":
        return Ur3eEnvRew0_05(urdf_ur3e_path, "UR3e", [0, 0, 0])
    elif env_name == "Ur3eTargetOrientation":
        return Ur3eTargetOrientation(urdf_ur3e_path, "UR3e", [0.2, 0.3, 0.25], [0, 0, 0, 1], evaluation_mode=True, use_gui=True)
    elif env_name == "Ur3eCurriculum":
        return Ur3eCurriculum(urdf_ur3e_path, "UR3e", [0.2, 0.3, 0.25], [0, 0, 0, 1], evaluation_mode=True, use_gui=True)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment to use")
    parser.add_argument("--model", type=str, required=True, help="Model path to save/load")
    parser.add_argument("--episodes", type=int, default=100, help="Number of testing episodes")
    args = parser.parse_args()

    urdf_ur3e_path = "../urdf/ur3e.urdf"
    ur3e_env = get_env(args.env, urdf_ur3e_path)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model for testing
    model = SAC.load("../policies/" +  args.model, device=device)

    # Testing loop
    n_episodes = args.episodes
    reward_threshold = 0
    max_steps = 500

    reward_above_threshold_count = 0
    episode_less_than_max_steps_count = 0

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}")
        obs, _ = ur3e_env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action, _ = model.predict(obs)
            obs, reward, done, _, info = ur3e_env.step(action)
            episode_reward += reward
            if done:
                if episode_reward > reward_threshold:
                    reward_above_threshold_count += 1
                if step < max_steps - 1:
                    episode_less_than_max_steps_count += 1
                break

    print(f"Number of episodes with reward > {reward_threshold}: {reward_above_threshold_count}, frequency: {reward_above_threshold_count / n_episodes:.3f}")
    print(f"Number of episodes with less than {max_steps} steps: {episode_less_than_max_steps_count}, frequency: {episode_less_than_max_steps_count / n_episodes:.3f}")

    # Close the environment
    ur3e_env.close()
