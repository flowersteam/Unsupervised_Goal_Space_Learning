#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains simulated environments.

Author: Alexandre Péré

"""

import numpy as np
import pygame
import pygame.gfxdraw
import itertools
import copy
import sys
if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


class BaseSimulation(object):
    """
    This defines the basic simulated environment methods.
    """

    def __init__(self):
        """
        The initializer of the BasicSimulation object.
        """

        # Initialize super
        object.__init__(self)

        # We initialize arguments
        self._factors = np.array([])
        self._factors_labels = list()
        self._factors_dims = list()
        self._params = {"width": 10,
                        "height": 10,
                        "depth": 1}
        self._rendering_vars = {}

    def _init_rendering(self):
        """
        This method is called to initialize the rendering engine.
        """

        raise NotImplementedError("Calling Virtual Method")

    def _render(self, fact_coord, depth=1):
        """
        The workhorse method. Renders the environment at coordinates in factors space.

        Args:
            fact_coord: a numpy array of size (nb_params) containing coordinates.
            depth: an int containing the depth of image.

        Returns:
            A numpy array of size (height, width (,depth)) containing simulation.
        """

        raise NotImplementedError("Calling Virtual Method")

    def _end_rendering(self):
        """
        This method is called to quit the rendering engine.
        """

        raise NotImplementedError("Calling Virtual Method")

    def nb_params(self):
        """
        This method returns the number of parameters of the simulation

        Returns:
            A int containing the number of parameters
        """

        return self._factors.size

    def get_factor_labels(self):
        """
        This method returns the factors labels.

        Returns:
            A list containing the factor labels.
        """

        return copy.deepcopy(self._factors_labels)

    def get_factor_dims(self):
        """
        This method returns the factors dimensions. 0 for discrete, 1 for continuous axial, 2 for continuous circular.

        Returns:
            A list of factors dimensions.
        """

        return copy.deepcopy(self._factors_dims)

    def draw(self, fact_coord, depth=1):
        """
        This method allows to draw a single point in factors space

        Args:
            fact_coord: a numpy array of size (nb_params) containing coordinates.
            depth: an int with depth of image. If one the output is 2 dimensional, else it is 3 Dim.

        Returns:
            A numpy array of size (height, width (,depth)) containing simulation.
        """

        # We check we are not virtual
        if type(self) is BaseSimulation:
            raise NotImplementedError("Calling Virtual Method")

        # We check input
        assert type(fact_coord) is list
        assert np.shape(fact_coord) == self._factors.shape
        assert np.max(fact_coord) <= 1.
        assert np.min(fact_coord) >= 0.

        # We initialize rendering
        self._init_rendering()

        # We render the image:
        output = self._render(fact_coord, depth=depth)

        # We end rendering
        self._end_rendering()

        return output

    def sample_all_factors(self, nb_samples, depth=1):
        """
        This method allows to sample all factors with same density.

        Args:
            nb_samples: an integer containing the number of samples along each factors.
            depth: image depth

        Returns:
            A numpy array of size (nb_samples**nb_factors, width, height(,depth)) containing samples.

        """

        #  We check we are not virtual
        if type(self) is BaseSimulation:
            raise NotImplementedError("Calling Virtual Method")

        # We check input
        assert type(nb_samples) is int

        # We initialize rendering
        self._init_rendering()

        # We instantiate the output array
        output = np.zeros([nb_samples**self._factors.size,
                           self._params["height"],
                           self._params["width"],
                           depth]).squeeze()

        # We generate the factor space
        space = [np.linspace(0., 1.,nb_samples) for i in range(self._factors.size)]

        # We render the images
        for index, coordinates in tqdm(enumerate(itertools.product(*space)), desc="Generating data", total=nb_samples**self._factors.size):
            output[index] = self._render(coordinates, depth=depth)

        # We quit rendering
        self._end_rendering()

        return output

    def sample_trajectory(self, start_coord, end_coord, nb_samples, depth=1):
        """
        This method allows to sample along a certain trajectory in factors space.

        Args:
            start_coord: Start point of the trajectory in factors space
            end_coord: End point of the trajectory in factors space
            nb_samples: number of samples to generate
            depth: image depth

        Returns:
            A numpy array of size (nb_samples, width, height (, depth)) containing samples.
        """

        #  We check we are not virtual
        if type(self) is BaseSimulation:
            raise NotImplementedError("Calling Virtual Method")

        # We check input
        assert type(nb_samples) is int
        assert type(start_coord) is list
        assert np.shape(start_coord) == self._factors.shape
        assert np.max(start_coord) <= 1.
        assert np.min(start_coord) >= 0.
        assert type(end_coord) is list
        assert np.shape(end_coord) == self._factors.shape
        assert np.max(end_coord) <= 1.
        assert np.min(end_coord) >= 0.


        # We initialize rendering
        self._init_rendering()

        # We instantiate the output array
        output = np.zeros([nb_samples,
                           self._params["height"],
                           self._params["width"],
                           depth]).squeeze()

        # We generate the factor space increment
        space_incre = (np.array(end_coord) - np.array(start_coord))/nb_samples

        # We render the images
        for index in tqdm(range(nb_samples), desc="Generating data", total=nb_samples):
            output[index] = self._render(start_coord+index*space_incre, depth)

        # We quit rendering
        self._end_rendering()

        return output


class RotatingArrowSimulation(BaseSimulation):
    """
    This class implement the single arm simulation.

    Factors:
        1. angle of rotation of the arrow
    """

    def __init__(self):

        # Initialize super
        BaseSimulation.__init__(self)

        # We initialize arguments
        self._factors = np.array([1.])
        self._factors_labels=["Rotation Angle"]
        self._factors_dims = [2]
        self._params = {"width": 70,
                        "height": 70,
                        "background_color":"black",
                        "arrow_color":"white",
                        "arrow_width":2,
                        "arrow_length":20}

        # We initialize the rendering environment
        pygame.init()
        self._rendering_vars = {"pygame_screen":None}

    def _init_rendering(self):

        # We generate pygame screen

        self._rendering_vars["pygame_screen"] = pygame.display.set_mode((self._params["width"], self._params["height"]),0,32)

    def _render(self, fact_coord, depth=1):

        # We compute real simulation parameters
        rotation_angle = 2.*np.pi*fact_coord[0]
        line_sp = (self._params["height"]//2, self._params["width"]//2)
        line_ep = (line_sp[0]+np.cos(rotation_angle)*self._params["arrow_length"],
                   line_sp[1]+np.sin(rotation_angle)*self._params["arrow_length"])

        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.draw.aaline(self._rendering_vars["pygame_screen"],
                         pygame.Color(self._params["arrow_color"]),
                         line_sp,
                         line_ep,
                         self._params["arrow_width"])
        pygame.display.update()

        # We retrieve the image as array
        if depth == 1:
            output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])
        elif depth == 3:
            output = pygame.surfarray.array3d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _end_rendering(self):

        # We quit pygame display
        pygame.display.quit()


class SingleJointArmSimulation(BaseSimulation):
    """
    This class implement the single joint arm simulation.

    Factors:
        1. angle of rotation of the first part of the arm
        2. angle of rotation of the second part of the arm (from the first)
    """

    def __init__(self):

        # Initialize super
        BaseSimulation.__init__(self)

        # We initialize arguments
        self._factors = np.array([1., 1.])
        self._factors_labels = ["Rotation Angle 1",
                               "Rotation Angle 2"]
        self._factors_dims = [2, 2]
        self._params = {"width": 70,
                        "height": 70,
                        "background_color": "black",
                        "arm_color": "white",
                        "arm_width": 2,
                        "arm_1_length": 20,
                        "arm_2_length": 13}

        # We initialize the rendering environment
        pygame.init()
        self._rendering_vars = {"pygame_screen": None}

    def _init_rendering(self):

        # We generate pygame screen
        self._rendering_vars["pygame_screen"] = pygame.display.set_mode((self._params["width"], self._params["height"]),0,32)

    def _render(self, fact_coord, depth=1):

        # We compute real simulation parameters
        angle_1 = 2*np.pi*fact_coord[0]
        angle_2 = 2*np.pi*fact_coord[1]
        line_1_sp = (self._params["height"]//2, self._params["width"]//2)
        line_1_ep = (line_1_sp[0]+np.cos(angle_1)*self._params["arm_1_length"],
                     line_1_sp[1]+np.sin(angle_1)*self._params["arm_1_length"])
        line_2_sp = line_1_ep
        line_2_ep = (line_2_sp[0]+np.cos(angle_1+angle_2)*self._params["arm_2_length"],
                     line_2_sp[1]+np.sin(angle_1+angle_2)*self._params["arm_2_length"])

        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.draw.aaline(self._rendering_vars["pygame_screen"],
                         pygame.Color(self._params["arm_color"]),
                         line_1_sp,
                         line_1_ep,
                         self._params["arm_width"])
        pygame.draw.aaline(self._rendering_vars["pygame_screen"],
                         pygame.Color(self._params["arm_color"]),
                         line_2_sp,
                         line_2_ep,
                         self._params["arm_width"])
        pygame.display.update()

        # We retrieve the image as array
        if depth == 1:
            output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])
        elif depth == 3:
            output = pygame.surfarray.array3d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _end_rendering(self):

        # We quit pygame display
        pygame.display.quit()


class TranslatingBallSimulation(BaseSimulation):
    """
    This class implement the translating ball simulation.

    Factors:
        1. x translation
        2. y translation
    """

    def __init__(self):

        # Initialize super
        BaseSimulation.__init__(self)

        # We initialize arguments
        self._factors = np.array([1., 1.])
        self._factors_labels = ["X Translation",
                               "Y Translation"]
        self._factors_dims = [1, 1]
        self._params = {"width": 70,
                        "height": 70,
                        "background_color": "black",
                        "ball_color": "white",
                        "ball_radius": 5}

        # We initialize the rendering environment
        pygame.init()
        self._rendering_vars = {"pygame_screen": None}

    def _init_rendering(self):

        # We generate pygame screen
        self._rendering_vars["pygame_screen"] = pygame.display.set_mode((self._params["width"], self._params["height"]),0,32)

    def _render(self, fact_coord, depth=1):

        # We compute real simulation parameters
        x = int(fact_coord[0]*self._params["width"])
        y = int(fact_coord[1]*self._params["height"])

        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.draw.circle(self._rendering_vars["pygame_screen"],
                           pygame.Color(self._params["ball_color"]),
                           (x,y),
                           self._params["ball_radius"],
                           0)
        pygame.display.update()

        # We retrieve the image as array
        if depth == 1:
            output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])
        elif depth == 3:
            output = pygame.surfarray.array3d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _end_rendering(self):

        # We quit pygame display
        pygame.display.quit()


class ArmAndChangingColorBall(BaseSimulation):
    """
    This class implement a simulation containing a single joint arm and a translating ball with changing color.

    Factors:
        1. x ball translation (continuous axial)
        2. y ball translation (continuous axial)
        3. ball color (discrete)
        4. arm angle 1 (continuous cyclic)
        5. arm angle 2 (continuous cyclic)
    """

    def __init__(self):

        # Initialize super
        BaseSimulation.__init__(self)

        # We initialize arguments
        self._factors = np.array([1., 1., 1., 1., 1.])
        self._factors_labels = ["X ball translation",
                               "Y ball translation",
                               "Ball Color",
                               "Arm angle 1",
                               "Arm angle 2"]
        self._factors_dims = [1, 1, 0, 2, 2]
        self._params = {"width": 224,
                        "height": 224,
                        "depth": 3,
                        "background_color": "black",
                        "arm_color": "white",
                        "arm_width": 2,
                        "arm_1_length": 30,
                        "arm_2_length": 20,
                        "ball_radius": 7}

        # We initialize the rendering environment
        pygame.init()
        self._rendering_vars = {"pygame_screen": None}

    def _init_rendering(self):

        # We generate pygame screen
        self._rendering_vars["pygame_screen"] = pygame.display.set_mode((self._params["width"], self._params["height"]),0,32)

    def _render(self, fact_coord, depth=1):

        # We compute real simulation parameters
        angle_1 = 2*np.pi*fact_coord[3]
        angle_2 = 2*np.pi*fact_coord[4]
        line_1_sp = (self._params["height"]//2, self._params["width"]//2)
        line_1_ep = (line_1_sp[0]+np.cos(angle_1)*self._params["arm_1_length"],
                     line_1_sp[1]+np.sin(angle_1)*self._params["arm_1_length"])
        line_2_sp = line_1_ep
        line_2_ep = (line_2_sp[0]+np.cos(angle_1+angle_2)*self._params["arm_2_length"],
                     line_2_sp[1]+np.sin(angle_1+angle_2)*self._params["arm_2_length"])
        x = int(fact_coord[0]*self._params["width"])
        y = int(fact_coord[1]*self._params["height"])
        c = "purple" if fact_coord[2] > 0.5 else "orange"

        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.draw.aaline(self._rendering_vars["pygame_screen"],
                         pygame.Color(self._params["arm_color"]),
                         line_1_sp,
                         line_1_ep,
                         self._params["arm_width"])
        pygame.draw.aaline(self._rendering_vars["pygame_screen"],
                         pygame.Color(self._params["arm_color"]),
                         line_2_sp,
                         line_2_ep,
                         self._params["arm_width"])
        pygame.draw.circle(self._rendering_vars["pygame_screen"],
                           pygame.Color(c),
                           (x,y),
                           self._params["ball_radius"],
                           0)
        pygame.display.update()

        # We retrieve the image as array
        if depth == 1:
            output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])
        elif depth == 3:
            output = pygame.surfarray.array3d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _end_rendering(self):

        # We quit pygame display
        pygame.display.quit()
