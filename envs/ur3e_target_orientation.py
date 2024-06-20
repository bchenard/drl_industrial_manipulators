import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3.common.monitor import Monitor

class Ur3eTargetOrientation(gym.Env):
    def __init__(self, urdf_file_path, robot_type, target_position, target_orientation):
        super(Ur3eTargetOrientation, self).__init__()
        self.urdf_file_path = urdf_file_path
        self.robot_type = robot_type
        self.physics_client = p.connect(p.DIRECT)  # Ensure only one connection is made
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # to get the plane.urdf
        self.init_state()
        self.target_position = target_position
        self.target_orientation = target_orientation
        self.step_counter = 0
        self._create_target_visual()

    def init_state(self):
        p.resetSimulation()
        self.robot = p.loadURDF(self.urdf_file_path, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])
        self.plane = p.loadURDF("plane.urdf")
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_state = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_joints,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints * 2 + 3 + 4,))  # 3 for position difference, 4 for quaternion

    def _create_target_visual(self):
        # Create a small sphere to represent the target position and orientation
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.05)
        self.target_visual = p.createMultiBody(baseMass=0,
                                               baseCollisionShapeIndex=collision_shape_id,
                                               baseVisualShapeIndex=visual_shape_id,
                                               basePosition=self.target_position,
                                               baseOrientation=self.target_orientation)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        p.resetSimulation()
        self.init_state()
        
        # Introduce randomness in target position and orientation
        self.target_position = np.random.uniform(low=[0.1, 0.1, 0.1], high=[0.5, 0.5, 0.5])
        random_euler_angles = np.random.uniform(low=[-np.pi, -np.pi, -np.pi], high=[np.pi, np.pi, np.pi])
        self.target_orientation = p.getQuaternionFromEuler(random_euler_angles)
        
        self._create_target_visual()
        self.step_counter = 0
        observation = self.get_observation()
        return observation, {}  # Return observation and an empty dictionary

    def get_observation(self):
        joint_states = [p.getJointState(self.robot, i) for i in range(self.num_joints)]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        tip_position, tip_orientation = p.getLinkState(self.robot, self.num_joints - 1)[:2]
        position_difference = np.array(tip_position) - np.array(self.target_position)
        
        # Calculate the relative quaternion
        relative_quaternion = p.getDifferenceQuaternion(tip_orientation, self.target_orientation)
        
        observation = np.concatenate([joint_positions, joint_velocities, position_difference, relative_quaternion])
        return observation

    def get_orientation_difference(self, current_orientation, target_orientation):
        # Calculate the relative quaternion
        relative_quaternion = p.getDifferenceQuaternion(current_orientation, target_orientation)
        
        # Sum of the absolute values of the quaternion components (excluding the norm)
        difference_sum = np.sum(np.abs(relative_quaternion))
        
        return difference_sum

    def get_reward(self, position_difference, current_orientation):
        position_error = np.linalg.norm(position_difference)
        
        # Calculate orientation error as the sum of the quaternion components
        orientation_error = self.get_orientation_difference(current_orientation, self.target_orientation)
        
        reward = - (position_error + orientation_error)  # Negative reward proportional to the error
        if position_error < 0.05 and orientation_error < 0.1:
            reward += 1000  # Bonus for reaching close to the target
        return reward

    def step(self, action):
        self.step_counter += 1
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=list(range(self.num_joints)),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=action)
        p.stepSimulation()
        
        self.joint_state = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        observation = self.get_observation()
        
        tip_position, tip_orientation = p.getLinkState(self.robot, self.num_joints - 1)[:2]
        position_difference = np.array(tip_position) - np.array(self.target_position)
        
        # Reward function
        reward = self.get_reward(position_difference, tip_orientation)
        
        done = False
        if self.step_counter >= 500:
            done = True
        
        info = {}
        terminated = done  # Use the same done flag for both terminated and truncated for simplicity
        truncated = False  # Not using truncation in this example
        
        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect()
