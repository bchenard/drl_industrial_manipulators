from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import torch
import argparse
import sys
import os

# Add parent directory to the sys.path
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
        return Ur3eTargetOrientation(urdf_ur3e_path, "UR3e", [0.2, 0.3, 0.25], [0, 0, 0, 1])
    elif env_name == "Ur3eCurriculum":
        return Ur3eCurriculum(urdf_ur3e_path, "UR3e", [0.2, 0.3, 0.25], [0, 0, 0, 1])
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def get_policy(policy_name, env, learning_rate, batch_size, device):
    if policy_name.lower() == "sac":
        return SAC("MlpPolicy", env, verbose=0, seed=0, device=device, learning_rate=learning_rate, batch_size=batch_size)
    elif policy_name.lower() == "ppo":
        return PPO("MlpPolicy", env, verbose=0, seed=0, device=device, learning_rate=learning_rate, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment to use")
    parser.add_argument("--model", type=str, required=True, help="Model path to save/load")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--training_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--policy", type=str, required=True, help="Policy to use (sac, ppo, ...)")
    args = parser.parse_args()

    urdf_ur3e_path = "../urdf/ur3e.urdf"
    ur3e_env = get_env(args.env, urdf_ur3e_path)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Wrap the environment in a monitor
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    model_name = os.path.splitext(args.model)[0]  # Extract the model name without the extension
    log_path = os.path.join(log_dir, model_name)
    monitored_env = Monitor(ur3e_env, filename=os.path.join(log_path, 'monitor.csv'))

    # Load or create the model
    model_path = os.path.join("../policies", args.model)
    try:
        model = SAC.load(model_path, device=device)
        model.set_env(monitored_env)
    except FileNotFoundError:
        print("Model not found, creating a new one")
        model = get_policy(args.policy, monitored_env, args.learning_rate, args.batch_size, device)

    # Configure TensorBoard logging
    model.set_logger(configure(log_path, ["tensorboard"]))

    # Train the model
    nb_steps = args.training_steps
    model.learn(nb_steps)

    # Save the model
    print("Model successfully saved to:", model_path)
    model.save(model_path)

    # Close the environment
    ur3e_env.close()
