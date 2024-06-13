import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from time import sleep

class Ur3eEnvRew0_01(gym.Env):
    def __init__(self, urdf_file_path, robot_type, targetPosition):
        super(Ur3eEnvRew0_01, self).__init__()
        self.urdf_file_path = urdf_file_path
        self.robot_type = robot_type
        self.physics_client = p.connect(p.DIRECT)  # Use GUI mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # to get the plane.urdf
        self.init_state()
        self.target_position = targetPosition
        self.step_counter = 0
        self.target_visual_shape_id = None
    
    def init_state(self):
        p.resetSimulation()
        self.robot = p.loadURDF(self.urdf_file_path, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])
        self.plane = p.loadURDF("plane.urdf")
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_state = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        self.action_space = gym.spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints * 2 + 3,))

    def reset(self, seed=0, options=None):
        super().reset(seed=seed, options=options)
        p.resetSimulation()
        self.init_state()
        self.step_counter = 0
        
        # Randomize initial position of the robot arm with valid positions
        for i in range(self.num_joints):
            lower_limit, upper_limit = p.getJointInfo(self.robot, i)[8:10]  # Get joint limits
            valid_position = False
            while not valid_position:
                joint_position = np.random.uniform(lower_limit, upper_limit)
                p.resetJointState(self.robot, i, joint_position)
                valid_position = self.check_joint_positions_valid()

        
        # Randomize the target position
        self.target_position = np.random.uniform([-0.5, -0.5, 0.1], [0.5, 0.5, 0.5]).tolist()
        
        # Add target visualization
        self.add_target_visualization()

        # sleep(1.5)
        
        observation = self.get_observation()
        return observation, {}

    def check_joint_positions_valid(self):
        tip_position, _ = p.getLinkState(self.robot, 7)[:2]
        return tip_position[2] > 0  # Ensure the tip is above the ground

    def add_target_visualization(self):
        if self.target_visual_shape_id is not None:
            p.removeBody(self.target_visual_shape_id)

        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        self.target_visual_shape_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=self.target_position)
    
    def get_observation(self):
        joint_states = [p.getJointState(self.robot, i) for i in range(self.num_joints)]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        tip_position, _ = p.getLinkState(self.robot, 7)[:2]
        difference = np.array(tip_position) - np.array(self.target_position)
        return np.concatenate([joint_positions, joint_velocities, difference])
    
    def get_reward(self, distance):
        if distance < 0.01:
            return 1000 - distance
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
        
        if reward > 99 or self.step_counter >= 500:
            done = True
            # sleep(1)
        else:
            done = False
            
        info = {}
        return observation, reward, done, False, info

    def close(self):
        p.disconnect()