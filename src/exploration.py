#!/usr/bin/python
# coding: utf-8
# Exploration

"""
This module contains exploration model classes.

Author: Alexandre Péré
"""

import explauto.sensorimotor_model.non_parametric
import explauto.utils.config
import numpy as np

class BaseExploration(object):
    """
    This class contains the exploration logic used.
    """

    def __init__(self, environment, explo_ratio, emb_size, transform_method, sampling_method, callback_period=100):
        """
        The initializer of the object.

        Args:
            + environment: the explauto environment on which the exploration is performed.
            + explo_ratio: the exploration ratio used in sensorimotor model
            + emb_size: the size of the latent space
            + transform_method: the method to call to embedd the data
            + sampling_method: the method to call to sample a goal
            + callback_method: the callback method called at given callback period
        """

        # We initialize the object
        object.__init__(self)
        # We set the environment
        self._environment = environment
        # We set the transform sampling and callback methods
        self._transform = transform_method
        self._sample = sampling_method
        self._callback = lambda: None
        self._callback_period = callback_period
        # We initialize the inner non parametric model
        conf = explauto.utils.config.make_configuration(self._environment.conf.m_mins,
                                                 self._environment.conf.m_maxs,
                                                 np.zeros(emb_size),
                                                 np.ones(emb_size))
        self._model = explauto.sensorimotor_model.non_parametric.NonParametric(conf, sigma_explo_ratio=explo_ratio, fwd='NN', inv='NN')
        # We initialize the curriculum lists
        self._state_list = list()
        self._latent_list = list()


    def explore(self, nb_iterations):
        """
        This method allows to run the exploration iterations.

        Args:
            + nb_iterations: the number of exploration iterations to run
        """

        pass

    def exploit(self, goal):
        """
        This method exploit the inverse model to perform a command that as close as possible to the goal.

        Args:
            + goal: the goal to achieve in state space
        """

        # We move in exploit mode
        self._model.mode = "exploit"
        # We send goal to latent space
        latent_goal = self._transform(goal)
        m = self._model.inverse_prediction(latent_goal)
        s = self._environment.update(m)
        # We reset explore mode
        self._model.mode = "explore"

        return s

    def get_states_limits(self):
        """
        This method allows to gather the limits of the explored state space.
        """

        return self._environment.conf.s_mins, self._environment.conf.s_maxs

    def get_reached_latents(self):
        """
        This method allows to gather the reached latent locations during exploration.
        """

        return self._latent_list

    def get_reached_states(self):
        """
        This method allows to gather the reached states locations during exploration.
        """

        return self._state_list

    def set_callback(self, callback_method):
        """
        This method allows to set the callback method.
        """

        self._callback = callback_method

class GoalBabblingExploration(BaseExploration):
    """
    This class contains the logic for goal babbling.
    """

    def explore(self, nb_iterations):

        # We set mode to
        self._model.mode = "explore"

        # Warm-Up
        m = self._environment.random_motors()[0]
        s = self._environment.update(m)
        z = self._transform(s)
        self._state_list += [s]
        self._latent_list += [z]
        self._model.update(m, z)

        # We loop through iterations
        for i in range(nb_iterations):
            # Random Motor Exploration + BootStrap
            bootstrap = np.logical_and.reduce(np.array(self._state_list)!=np.array(self._environment.initial_pose), axis=1).sum() == 0
            if bootstrap or i < 100 or np.random.random() < 0.2:
                m = self._environment.random_motors()[0]
            # Goal Babbling
            else:
                # Change here the sampling logic
                z_goal = self._sample()
                m = self._model.inverse_prediction(z_goal)
            # We perform the motor action
            s = self._environment.update(m)
            # We embedd the reached state
            z = self._transform(s)
            # We compute the latent dimension
            self._state_list += [s] # ball positions at the end of movements
            self._latent_list += [z] # latent representation of ball position
            # We update the model
            self._model.update(m, z) # update sensorimotor model
            # We call the callback method
            if i%self._callback_period == 0 and i > 0:
                self._callback()

class RandomMotorBabblingExploration(BaseExploration):
    """
    This class contains the logic for random motor babbling.
    """

    def explore(self, nb_iterations):

        # We set mode to
        self._model.mode = "explore"

        # We loop through iterations
        for i in range(nb_iterations):
            # Random Babbling
            m = self._environment.random_motors()[0]
            # We perform the motor action
            s = self._environment.update(m)
            # We compute the latent dimension
            self._state_list += [s] # ball positions at the end of movements
            # We update the model
            self._model.update(m, s) # update sensorimotor model
            # We call the callback method
            if i%self._callback_period == 0 and i > 0:
                self._callback()
