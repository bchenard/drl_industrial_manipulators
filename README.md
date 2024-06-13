# drl_industrial_manipulators
Exploring Deep Reinforcement Learning (DRL) methods for the programmation of industrial maniuplators

To generate documentation for executing the provided scripts (`evaluation/main.py` and `training/main.py`), we can create a simple README file that outlines how to run each script along with the necessary arguments. Here's an example of what the documentation might look like:

---

## Script Documentation

### 1. Evaluation Script (`evaluation/main.py`)

This script is used to evaluate a pre-trained reinforcement learning model on a specified environment.

#### Usage:

```bash
python3 evaluation/main.py --env <environment_name> --model <model_name> [--episodes <number_of_episodes>]
```

#### Arguments:

- `--env`: Specify the environment to use for evaluation (e.g., "Ur3eEnvRew0_01", "Ur3eEnvRew0_05").
- `--model`: Path to the pre-trained model to load for evaluation.
- `--episodes` (optional): Number of episodes to run the evaluation (default is 100).

#### Example:

```bash
python3 evaluation/main.py --env Ur3eEnvRew0_01 --model my_sac_model --episodes 50
```

### 2. Training Script (`training/main.py`)

This script is used to train a reinforcement learning model on a specified environment.

#### Usage:

```bash
python3 training/main.py --env <environment_name> --model <model_name> --policy <policy_name> [--learning_rate <lr>] [--batch_size <batch_size>] [--training_steps <training_steps>]
```

#### Arguments:

- `--env`: Specify the environment to use for training (e.g., "Ur3eEnvRew0_01", "Ur3eEnvRew0_05").
- `--model`: Name of the model to save or load (will be saved in '../policies/' directory).
- `--policy`: Specify the RL policy to use for training ("sac" for Soft Actor-Critic, "ppo" for Proximal Policy Optimization, etc.).
- `--learning_rate` (optional): Learning rate for the training process (default is 0.001).
- `--batch_size` (optional): Batch size for the training process (default is 64).
- `--training_steps` (optional): Number of training steps to perform (default is 10000).

#### Example:

```bash
python3 training/main.py --env Ur3eEnvRew0_05 --model my_sac_model --policy sac --learning_rate 0.0005 --batch_size 128 --training_steps 20000
```

---

### Notes:

- Ensure that Python environment includes all necessary dependencies, including `stable_baselines3`,  `torch` and `gymnasium`.
- CUDA will be used if available; otherwise, the CPU will be used for computation.

This documentation provides clear instructions on how to execute both scripts with examples of typical usage scenarios. Adjust the examples and paths based on your specific project setup and requirements.