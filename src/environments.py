# coding: utf-8
# Armball Env

"""
This module contains simulation logic for different environment.

Author: Alexandre Péré (with help from Sebastien Forestier)
"""

import numpy as np
from explauto.environment.dynamic_environment import DynamicEnvironment
from explauto.environment.modular_environment import FlatEnvironment, HierarchicalEnvironment
from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment
from explauto.environment.simple_arm.simple_arm import joint_positions
from explauto.utils.utils import rand_bounds


class ArmBallStatic(Environment):
    """
    This environment defines the Static ArmBall problem.
    """

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 arm_lengths, arm_angle_shift, arm_rest_state,
                 ball_size, ball_initial_position):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.arm_lengths = arm_lengths
        self.arm_angle_shift = arm_angle_shift
        self.arm_rest_state = arm_rest_state
        self.ball_size = ball_size
        self.size_sq = ball_size * ball_size
        self.ball_initial_position = ball_initial_position
        self.trajectory = list()
        self.reset()

    def reset(self):

        # We reset the motion state of the ball
        self.ball_move = False
        # We reset the position of the ball
        self.ball_pos = np.array(self.ball_initial_position)
        # We reset the list
        self.trajectory = list()

    def compute_motor_command(self, m):

        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):

        # We compute the position and angle of the end effector
        a = self.arm_angle_shift + np.cumsum(np.array(m))
        a_pi = np.pi * a
        hand_pos = np.array([np.sum(np.cos(a_pi)*self.arm_lengths), np.sum(np.sin(a_pi)*self.arm_lengths)])
        angle = np.mod(a[-1] + 1, 2) - 1

        # We move the ball
        if self.ball_move or ((hand_pos[0] - self.ball_pos[0]) ** 2 + (hand_pos[1] - self.ball_pos[1]) ** 2 < self.size_sq):
            self.ball_pos = hand_pos
            self.ball_move = 1

        # We append the trajectory
        self.trajectory.append(self.ball_pos)

        return list(self.ball_pos)

class ArmBallDynamic(DynamicEnvironment):
    """
    This dynamic environment defines the dynamic logic of ArmBall problem
    """

    def __init__(self, arm_ball_static_config, n_dmp_basis=3, goal_size=1.):

        n_joints = len(arm_ball_static_config["m_mins"])

        dynamic_environment_config = dict(
            env_cfg=arm_ball_static_config,
            env_cls=ArmBallStatic,
            m_mins=[-1.] * n_joints * n_dmp_basis,
            m_maxs=[1.] * n_joints * n_dmp_basis,
            s_mins=[-goal_size] * 2,
            s_maxs=[goal_size] * 2,
            n_bfs=n_dmp_basis,
            move_steps=50,
            n_dynamic_motor_dims=n_joints,
            n_dynamic_sensori_dims=2,
            sensori_traj_type="end_point",
            max_params=1000)

        DynamicEnvironment.__init__(self, **dynamic_environment_config)

        self.initial_pose = arm_ball_static_config['ball_initial_position']

    def get_last_traj(self):

        return np.array(self.env.trajectory)


class ArmTwoBallsStatic(Environment):
    """
    This environment defines the Static Arm with 2 balls problem.
    """

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 arm_lengths, arm_angle_shift, arm_rest_state,
                 ball_size, balls_initial_position):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.arm_lengths = arm_lengths
        self.arm_angle_shift = arm_angle_shift
        self.arm_rest_state = arm_rest_state
        self.ball_size = ball_size
        self.size_sq = ball_size * ball_size
        self.balls_initial_position = balls_initial_position
        self.initial_pose = balls_initial_position
        self._trajectory=list()
        self.reset()

    def reset(self):

        # We reset the motion state of the balls
        self.ball_1_move = False
        self.ball_2_move = False
        # We reset the position of the balls
        self.ball_1_pos = np.array(self.balls_initial_position[:2])
        self.ball_2_pos = np.array(self.balls_initial_position[2:])
        # We reset the trajectory
        self.trajectory = list()

    def compute_motor_command(self, m):
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):

        # We compute the position of the end effector and the angle of it
        a = self.arm_angle_shift + np.cumsum(np.array(m))
        a_pi = np.pi * a
        hand_pos = np.array([np.sum(np.cos(a_pi)*self.arm_lengths), np.sum(np.sin(a_pi)*self.arm_lengths)])
        angle = np.mod(a[-1] + 1, 2) - 1

        # We move the first ball
        cond_grip_1 =  ((hand_pos[0] - self.ball_1_pos[0]) ** 2 + (hand_pos[1] - self.ball_1_pos[1]) ** 2 < self.size_sq) and not self.ball_2_move
        if self.ball_1_move or cond_grip_1:
            self.ball_1_pos = hand_pos
            self.ball_1_move = True

        # We move the second ball
        cond_grip_2 =  ((hand_pos[0] - self.ball_2_pos[0]) ** 2 + (hand_pos[1] - self.ball_2_pos[1]) ** 2 < self.size_sq) and not self.ball_1_move
        if self.ball_2_move or cond_grip_2:
            self.ball_2_pos = hand_pos
            self.ball_2_move = True

        # We append the point
        effect = np.concatenate([self.ball_1_pos,self.ball_2_pos])
        self.trajectory.append(effect)

        return effect.tolist()


class ArmTwoBallsDynamic(DynamicEnvironment):
    """
    This environment defines the Dynamic of the Arm with 2 balls environment.
    """

    def __init__(self, arm_two_balls_static_config, n_dmp_basis=3, goal_size=1.):

        n_joints = len(arm_two_balls_static_config["m_mins"])

        dynamic_environment_config = dict(
            env_cfg=arm_two_balls_static_config,
            env_cls=ArmTwoBallsStatic,
            m_mins=[-1.] * n_joints * n_dmp_basis,
            m_maxs=[1.] * n_joints * n_dmp_basis,
            s_mins=[-goal_size] * 4,
            s_maxs=[goal_size] * 4,
            n_bfs=n_dmp_basis,
            move_steps=50,
            n_dynamic_motor_dims=n_joints,
            n_dynamic_sensori_dims=2,
            sensori_traj_type="end_point",
            max_params=1000)

        DynamicEnvironment.__init__(self, **dynamic_environment_config)

        self.initial_pose = arm_two_balls_static_config['balls_initial_position']

    def get_last_traj(self):

        return np.array(self.env.trajectory)


class ArmArrowStatic(Environment):
    """
    This environment defines the Static Arm with arrow (orientable object) problem.
    """

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 arm_lengths, arm_angle_shift, arm_rest_state,
                 arrow_size, arrow_initial_pose):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.arm_lengths = arm_lengths
        self.arm_angle_shift = arm_angle_shift
        self.arm_rest_state = arm_rest_state
        self.arrow_size = arrow_size
        self.size_sq = arrow_size * arrow_size
        self.arrow_initial_pose = arrow_initial_pose
        self.initial_pose = arrow_initial_pose
        self.trajectory = list()
        self.reset()

    def reset(self):

        # We reset the motion state of the arrow
        self.arrow_move = False
        # We reset the pose of the arrow
        self.arrow_pos = np.array(self.arrow_initial_pose)
        # We reset the trajectory
        self.trajectory = list()

    def compute_motor_command(self, m):
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):

        # We compute the position of the end effector and the angle of it
        a = self.arm_angle_shift + np.cumsum(np.array(m))
        a_pi = np.pi * a
        hand_pos = np.array([np.sum(np.cos(a_pi)*self.arm_lengths), np.sum(np.sin(a_pi)*self.arm_lengths)])
        angle = np.mod(a[-1] + 1, 2) - 1
        angle_pi = a_pi[-1] + np.pi

        # We move the arrow
        cond_grip =  ((hand_pos[0] - self.arrow_pos[0]) ** 2 + (hand_pos[1] - self.arrow_pos[1]) ** 2 < self.size_sq)
        if self.arrow_move or cond_grip:
            self.arrow_pos = np.concatenate([hand_pos,[angle]])
            self.arrow_move = True

        # We append the trajectory
        effect = np.clip(self.arrow_pos, -1, 1)
        self.trajectory.append(effect.tolist())

        return effect.tolist()


class ArmArrowDynamic(DynamicEnvironment):
    """
    This dynamic environment defines the dynamic logic of ArmBall problem
    """

    def __init__(self, arm_arrow_static_config, n_dmp_basis=3, goal_size=1.):

        n_joints = len(arm_arrow_static_config["m_mins"])

        dynamic_environment_config = dict(
            env_cfg=arm_arrow_static_config,
            env_cls=ArmArrowStatic,
            m_mins=[-1.] * n_joints * n_dmp_basis,
            m_maxs=[1.] * n_joints * n_dmp_basis,
            s_mins=[-goal_size] * 3,
            s_maxs=[goal_size] * 3,
            n_bfs=n_dmp_basis,
            move_steps=50,
            n_dynamic_motor_dims=n_joints,
            n_dynamic_sensori_dims=2,
            sensori_traj_type="end_point",
            max_params=1000)

        DynamicEnvironment.__init__(self, **dynamic_environment_config)

        self.initial_pose = arm_arrow_static_config['arrow_initial_pose']

    def get_last_traj(self):

        return np.array(self.env.trajectory)


class ArmToolBallStatic(Environment):
    """
    This environment defines the Static Arm with tool and ball problem.
    """

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 arm_lengths, arm_angle_shift, arm_rest_state,
                 ball_size, ball_initial_position,
                 tool_length, tool_initial_pose):

        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.arm_lengths = arm_lengths
        self.arm_angle_shift = arm_angle_shift
        self.arm_rest_state = arm_rest_state
        self.ball_size = ball_size
        self.size_sq = ball_size * ball_size
        self.ball_initial_position = ball_initial_position
        self.tool_length = tool_length
        self.tool_initial_pose = tool_initial_pose
        self.initial_pose = np.concatenate([ball_initial_position, tool_initial_pose]).tolist()
        self.trajectory = list()
        self.reset()

    def reset(self):

        # We reset the motion state of the ball and tool
        self.ball_move = False
        self.tool_move = False
        # We reset the position of the ball and tool
        self.ball_pos = np.array(self.ball_initial_position)
        self.tool_pos = np.array(self.tool_initial_pose)
        # We reset the trajectory
        self.trajectory = list()

    def compute_motor_command(self, m):
        
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, m):

        # We compute the position of the end effector and the angle of it
        a = self.arm_angle_shift + np.cumsum(np.array(m))
        a_pi = np.pi * a
        hand_pos = np.array([np.sum(np.cos(a_pi)*self.arm_lengths), np.sum(np.sin(a_pi)*self.arm_lengths)])
        angle = np.mod(a[-1] + 1, 2) - 1
        angle_pi = a_pi[-1] + np.pi

        # We move the tool
        cond_grip_tool =  ((hand_pos[0] - self.tool_pos[0]) ** 2 + (hand_pos[1] - self.tool_pos[1]) ** 2 < self.size_sq)
        if self.tool_move or cond_grip_tool:
            self.tool_pos = np.concatenate([hand_pos, [angle]])
            self.tool_move = True
            # We compute the position of the end effector of the tool
            end_pos = np.array([hand_pos[0]+np.cos(angle_pi)*self.tool_length, hand_pos[1]+np.sin(angle_pi)*self.tool_length])
            # We move the ball
            cond_grip_ball =  ((end_pos[0] - self.ball_pos[0]) ** 2 + (end_pos[1] - self.ball_pos[1]) ** 2 < self.size_sq)
            if self.ball_move or cond_grip_ball:
                self.ball_pos = end_pos
                self.ball_move = True

        # We append the trajectory
        effect = np.concatenate([self.ball_pos,self.tool_pos])
        effect = np.clip(effect, -1, 1).tolist()
        self.trajectory.append(effect)

        return effect


class ArmToolBallDynamic(DynamicEnvironment):
    """
    This dynamic environment defines the dynamic logic of Arm tool Ball problem
    """

    def __init__(self, arm_tool_ball_static_config, n_dmp_basis=3, goal_size=1.):

        n_joints = len(arm_tool_ball_static_config["m_mins"])

        dynamic_environment_config = dict(
            env_cfg=arm_tool_ball_static_config,
            env_cls=ArmToolBallStatic,
            m_mins=[-1.] * n_joints * n_dmp_basis,
            m_maxs=[1.] * n_joints * n_dmp_basis,
            s_mins=[-goal_size] * 5,
            s_maxs=[goal_size] * 5,
            n_bfs=n_dmp_basis,
            move_steps=50,
            n_dynamic_motor_dims=n_joints,
            n_dynamic_sensori_dims=2,
            sensori_traj_type="end_point",
            max_params=1000)

        DynamicEnvironment.__init__(self, **dynamic_environment_config)

        self.initial_pose = arm_tool_ball_static_config['ball_initial_position'] + arm_tool_ball_static_config['tool_initial_pose']

    def get_last_traj(self):

        return np.array(self.env.trajectory)
