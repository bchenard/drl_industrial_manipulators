import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import sys
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
import sys
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

class Ur3eEnv(gym.Env):
    def __init__(self, urdf_file_path, robot_type, targetPosition):
        super(Ur3eEnv, self).__init__()
        self.urdf_file_path = urdf_file_path
        self.robot_type = robot_type
        self.physics_client = p.connect(p.DIRECT)  # Ensure only one connection is made
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # to get the plane.urdf
        self.init_state()
        self.target_position = targetPosition
        self.step_counter = 0
    
    def init_state(self):
        p.resetSimulation()
        self.robot = p.loadURDF(self.urdf_file_path, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])
        self.plane = p.loadURDF("plane.urdf")
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_state = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_joints,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints * 2 + 3,))

    def reset(self, seed=0, options=None):
        super().reset(seed=seed, options=options)
        p.resetSimulation()
        self.init_state()
        self.step_counter = 0
        observation = self.get_observation()
        return observation, {}

    def get_observation(self):
        joint_states = [p.getJointState(self.robot, i) for i in range(self.num_joints)]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        tip_position, _ = p.getLinkState(self.robot, 7)[:2]
        difference = np.array(tip_position) - np.array(self.target_position)
        return np.concatenate([joint_positions, joint_velocities, difference])
    
    def get_reward(self, distance):
        if distance < 0.05:
            return 100 - distance
        else:
            return -distance
        

    def step(self, action):
        self.step_counter += 1
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=list(range(self.num_joints)),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=action)
        p.stepSimulation()
        
        self.joint_state = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        observation = self.get_observation()
        
        tip_position, _ = p.getLinkState(self.robot, 7)[:2]
        
        # Calculate the distance to the target position
        distance = np.linalg.norm(np.array(tip_position) - np.array(self.target_position))

        # Reward function
        reward = self.get_reward(distance)
        
        done = False
        # Better performances without resetting (only one episode)
        if (self.step_counter >= 500):
            done = True
        info = {}
        return observation, reward, done, False, info

    def close(self):
        p.disconnect()

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the target position as command line arguments.")
        sys.exit(1)

    urdf_ur3e_path = "/home/bchenard/Documents/drl_env/stage/pybullet/ur3e/ur3e.urdf"
    ur3e_env = Ur3eEnv(urdf_ur3e_path, "UR3e", [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])])

    # Wrap the environment in a monitor
    filename = "logs_training/ppo_ur3e_"
    filename += "_".join([str(pos) for pos in ur3e_env.target_position])
    monitored_env = Monitor(ur3e_env, filename=filename)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    # model = PPO.load("/home/bchenard/Documents/drl_env/stage/pybullet/ur3e/models/ppo_ur3e_", device=device, verbose=1, env=monitored_env, seed=0)
    # model.learn(400_000)
    
    model = PPO("MlpPolicy", monitored_env, verbose=1, seed=0, device=device, learning_rate=0.0003, batch_size=256).learn(500_000)
    
    # Save the model
    model_path = "/home/bchenard/Documents/drl_env/stage/pybullet/ur3e/models/ppo_ur3e_"
    print("Model successfully saved to:", model_path)
    model.save(model_path)
    
    
    
    # Wrap the environment in a vectorized environment
    vec_env = DummyVecEnv([lambda: ur3e_env])
    obs = vec_env.reset()
    # Variables to track the best configuration
    best_obs = None
    best_action = None
    max_reward = -float('inf')
    best_step = 0

    for i in range(500):
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        
        # Calculate the distance to the target position
        tip_position, _ = p.getLinkState(ur3e_env.robot, 7)[:2]
        distance = np.linalg.norm(np.array(tip_position) - np.array(ur3e_env.target_position))
        

        # Update the best configuration if the current reward is better
        if rewards > max_reward:
            max_reward = rewards
            best_obs = obs
            best_action = action
            best_step = i

        # if distance < 0.005:
        #     print("Episode finished after {} timesteps".format(i), "Distance:", distance)
        #     break
        if done:
            print("Episode finished after {} timesteps".format(i), "Distance:", distance)
            break


    print("Best reward achieved: ", max_reward)
    print("Achieved at step: ", best_step)
    print("Best action: \n", best_action)
    print("Best observation: \n", best_obs)

    # Keep the simulator running
    input("Press Enter to close the simulator...")
    ur3e_env.close()
