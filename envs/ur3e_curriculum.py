import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3.common.monitor import Monitor
import time
from stable_baselines3.common.logger import configure

class Ur3eCurriculum(gym.Env):
    def __init__(self, urdf_file_path, robot_type, target_position, target_orientation, use_gui=False, evaluation_mode=False):
        super(Ur3eCurriculum, self).__init__()
        self.urdf_file_path = urdf_file_path
        self.robot_type = robot_type
        self.use_gui = use_gui
        self.evaluation_mode = evaluation_mode

        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.init_state()

        self.initial_target_position = target_position
        self.initial_target_orientation = target_orientation
        self.target_position = target_position
        self.target_orientation = target_orientation

        self.step_counter = 0
        self.curriculum_level = 0
        self.success_counter = 0
        self.success_threshold = 10

        self.reward_thresholds_pos = [0.6, 0.5, 0.35, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.008, 0.005, 0.003, 0.002]
        self.reward_thresholds_orient = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03]
        self.reward_thresholds = [1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        self.logger = configure("logs", ["tensorboard"])

        self._create_target_visual()

    def init_state(self):
        p.resetSimulation()
        self.robot = p.loadURDF(self.urdf_file_path, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])
        self.plane = p.loadURDF("plane.urdf")
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_state = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_joints,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_joints * 2 + 3 + 4,))

    def _create_target_visual(self):
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.05)
        self.target_visual = p.createMultiBody(baseMass=0,
                                               baseCollisionShapeIndex=collision_shape_id,
                                               baseVisualShapeIndex=visual_shape_id,
                                               basePosition=self.target_position,
                                               baseOrientation=self.target_orientation)

    def update_curriculum(self):
        if self.success_counter >= self.success_threshold and self.curriculum_level < len(self.reward_thresholds_pos) - 1:
            self.curriculum_level += 1
            self.success_counter = 0
            print(f"Leveling up! New curriculum level: {self.curriculum_level}")
            self.logger.record("curriculum/level", self.curriculum_level)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        p.resetSimulation()
        self.init_state()

        self.target_position = np.random.uniform(low=[0.1, 0.1, 0.1], high=[0.5, 0.5, 0.5])
        random_euler_angles = np.random.uniform(low=[-np.pi, -np.pi, -np.pi], high=[np.pi, np.pi, np.pi])
        self.target_orientation = p.getQuaternionFromEuler(random_euler_angles)

        self._create_target_visual()
        self.step_counter = 0
        observation = self.get_observation()
        return observation, {}

    def get_observation(self):
        joint_states = [p.getJointState(self.robot, i) for i in range(self.num_joints)]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        tip_position, tip_orientation = p.getLinkState(self.robot, self.num_joints - 1)[:2]
        position_difference = np.array(tip_position) - np.array(self.target_position)

        relative_quaternion = p.getDifferenceQuaternion(tip_orientation, self.target_orientation)

        observation = np.concatenate([joint_positions, joint_velocities, position_difference, relative_quaternion])
        return observation

    def get_orientation_difference(self, current_orientation, target_orientation):
        relative_quaternion = p.getDifferenceQuaternion(current_orientation, target_orientation)
        difference_norm = np.linalg.norm(relative_quaternion[1:])
        return difference_norm

    def get_reward(self, position_difference, current_orientation):
        position_error = np.linalg.norm(position_difference)
        orientation_error = self.get_orientation_difference(current_orientation, self.target_orientation)
        reward = 0
        threshold_pos = self.reward_thresholds_pos[self.curriculum_level]
        threshold_orient = self.reward_thresholds_orient[self.curriculum_level]
        # print(f"Position error: {position_error}, Orientation error: {orientation_error}")
        if (position_error + orientation_error) < self.reward_thresholds[self.curriculum_level]:
            reward = 1000 * (self.curriculum_level + 1)
            self.success_counter += 1
        else:
            self.success_counter = max(0, self.success_counter - 1)
        return reward

    def step(self, action):
        self.step_counter += 1
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=list(range(self.num_joints)),
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=action)
        p.stepSimulation()

        if self.evaluation_mode:
            time.sleep(0.01)

        self.joint_state = [p.getJointState(self.robot, i)[0] for i in range(self.num_joints)]
        observation = self.get_observation()

        tip_position, tip_orientation = p.getLinkState(self.robot, self.num_joints - 1)[:2]
        position_difference = np.array(tip_position) - np.array(self.target_position)

        reward = self.get_reward(position_difference, tip_orientation)

        done = False
        if self.step_counter >= 500:
            done = True
            self.update_curriculum()

        info = {}
        terminated = done
        truncated = False

        return observation, reward, terminated, truncated, info

    def close(self):
        p.disconnect()
