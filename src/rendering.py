#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains renderers for environments.

Author: Alexandre Péré

"""

import numpy as np
import pygame
import pygame.gfxdraw
import skimage.util
import skimage.transform
import skimage.exposure
import copy
import os
import array2gif

os.environ["SDL_VIDEODRIVER"] = "dummy"


class BaseRenderer(object):
    """
    This defines the basic rendering environment methods.
    """

    def __init__(self, noise=.0, distractor=False, deformation=.0, outliers=.0):
        """
        The initializer of the BasicRenderer object.
        """

        # Initialize super
        object.__init__(self)

        # We initialize arguments
        self._factors = np.array([])
        self._params = {"width": 70,
                        "height": 70,
                        'background_color': 'black',
                        'foreground_color': 'white',
                        'noise': noise,
                        'distractor': distractor,
                        'deformation': deformation,
                        'outliers': outliers}
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
            A numpy array of size (height, width) containing rendering.
        """

        raise NotImplementedError("Calling Virtual Method")

    def _render_outlier(self):
        """
        This method allows to render a random outlier (draw a random 5 points polygon)
        """

        # We generate 5 random points
        points = [[int(np.random.uniform(0, 1) * self._params["width"]),
                   int(np.random.uniform(0, 1) * self._params["height"])] for i in range(5)]

        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.gfxdraw.aapolygon(self._rendering_vars["pygame_screen"],points,pygame.Color(self._params["foreground_color"]))
        pygame.gfxdraw.filled_polygon(self._rendering_vars["pygame_screen"],points,pygame.Color(self._params["foreground_color"]))
        pygame.display.update()

        # We retrieve the image as array
        output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _render_distractor(self):
        """
        This method allows to render a distractor (ball) positioned randomly  on the image.
        """

        # We compute real rendering parameters
        x = int(np.random.uniform(0, 1) * self._params["width"])
        y = int(np.random.uniform(0, 1) * self._params["height"])

        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.gfxdraw.aacircle(self._rendering_vars["pygame_screen"],x,y,5,pygame.Color(self._params["foreground_color"]))
        pygame.gfxdraw.filled_circle(self._rendering_vars["pygame_screen"],x,y,5,pygame.Color(self._params["foreground_color"]))
        pygame.display.update()

        # We retrieve the image as array
        output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _end_rendering(self):
        """
        This method is called to quit the rendering engine.
        """

        raise NotImplementedError("Calling Virtual Method")

    def _sample_full(self):
        """
        This method allows to uniformly sample an image in the full state space.
        """

        raise NotImplementedError("Calling Virtual Method")

    def _sample_sub(self):
        """
        This method allows to uniformly sample an image in a convex subspace of the state space
        """

        raise NotImplementedError("Calling Virtual Method")

    def _sample_corr(self):
        """
        This method allows to uniformly sample an image in a hyperplane of the state space containing correlated situations.
        """

        raise NotImplementedError('Calling Virtual Method')

    def nb_params(self):
        """
        This method returns the number of parameters of the rendering

        Returns:
            A int containing the number of parameters
        """

        return self._factors.size

    def get_factor_dims(self):
        """
        This method returns the factors dimensions. 0 for discrete, 1 for continuous axial, 2 for continuous circular.

        Returns:
            A list of factors dimensions.
        """

        return copy.deepcopy(self._factors_dims)

    def draw(self, fact_coord):
        """
        This method allows to draw a single point in factors space.

        Args:
            fact_coord: a numpy array of size (nb_params) containing coordinates.

        Returns:
            A numpy array of size (height, width) containing rendering.
        """

        #  We check we are not virtual
        if type(self) is BaseRenderer: raise NotImplementedError("Calling Virtual Method")

        # We turn coords to list if necessary
        if type(fact_coord) is np.ndarray: fact_coord=fact_coord.tolist()

        # We check input
        assert np.shape(fact_coord) == self._factors.shape
        assert np.max(fact_coord) <= 1.
        assert np.min(fact_coord) >= 0.

        # We initialize rendering
        self._init_rendering()

        # We render the image:
        output = self._render(fact_coord)

        # We corrupt the image depending on the corruption parameters
        if self._params['noise'] > 0.:
            output = skimage.util.random_noise(output, mode='s&p', amount=self._params['noise'])
        if self._params['distractor']:
            distraction = self._render_distractor()
            output[np.where(distraction==1.)] = distraction[np.where(distraction==1.)]
        if self._params['deformation'] > 0.:
            p_rot = np.random.uniform(-1, 1) * 0.02 * self._params['deformation']  # Max 0.02 of rotation
            p_trans = [np.random.uniform(-1, 1) * self._params['width'] * 0.01 * self._params['deformation'],
                       # Max 1 pct
                       np.random.uniform(-1, 1) * self._params['height'] * 0.01 * self._params['deformation']]  # of trs
            p_shear = np.random.uniform(-1, 1) * 0.05 * self._params['deformation']  # Max 0.05 of shear
            trans = skimage.transform.AffineTransform(rotation=p_rot, translation=p_trans, shear=p_shear)
            output = skimage.transform.warp(output, trans.inverse)
        if self._params['outliers'] > 0. and np.random.uniform(0,1) < self._params['outliers']:
            output = self._render_outlier()

        # We end rendering
        self._end_rendering()

        return output

    def sample(self, nb_samples, type='full'):
        """
        This method allows to sample random images to train the representation. Sampling can follow different policies
        based on the type argument.

        Args:
            nb_samples: The number of samples to produce
            type: 'full' to sample uniformly the full state space
                  'sub' to sample a convex subspace of the state space
                  'corr' to sample a correlated subspace of the state space

        Returns:
            A numpy array containing the data.
        """

        # We initialize a running list
        output = list()

        # We fill the list
        for i in range(nb_samples):
            if type is 'full':
                output.append(self._sample_full())
            elif type is 'sub':
                output.append(self._sample_sub())
            elif type is 'corr':
                output.append(self._sample_corr())

        return np.array(output)
    
    def save_gif(self, trajectory, name):
        """
        This method allows to save a trajectory as a gif. 
        """

        # We instantiate dataset
        image_list = list()
        # We loop through coordinates to write images.
        for coord in trajectory:
            coord = (coord+1)/2
            arr = self.draw(coord.tolist()) * 255.0
            arr = arr.astype(np.uint8)
            arr = arr.reshape([1,70,70])
            arr = np.repeat(arr,3,axis=0)
            image_list.append(arr)

        array2gif.write_gif(image_list, name, fps=25)

class ArmBallRenderer(BaseRenderer):
    """
    This class implement the armball renderer

    Factors:
        1. x translation
        2. y translation
    """

    def __init__(self, **kwargs):
        # Initialize super
        BaseRenderer.__init__(self, **kwargs)

        # We initialize arguments
        self._factors = np.array([1., 1.])
        self._factors_dims = np.array([1, 1])
        self._params.update({"width": 70,
                             "height": 70,
                             "background_color": "black",
                             "ball_color": "white",
                             "ball_radius": 5})

        # We initialize the rendering environment
        pygame.init()
        self._rendering_vars = {"pygame_screen": None}

    def _init_rendering(self):
        # We generate pygame screen
        self._rendering_vars["pygame_screen"] = pygame.display.set_mode((self._params["width"], self._params["height"]),
                                                                        0, 32)

    def _render(self, fact_coord):
        # We compute real rendering parameters
        x = int(fact_coord[0] * self._params["width"])
        y = int(fact_coord[1] * self._params["height"])

        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.gfxdraw.aacircle(self._rendering_vars["pygame_screen"],x,y,self._params['ball_radius'],pygame.Color(self._params["ball_color"]))
        pygame.gfxdraw.filled_circle(self._rendering_vars["pygame_screen"],x,y,self._params['ball_radius'],pygame.Color(self._params["ball_color"]))
        pygame.display.update()

        # We retrieve the image as array
        output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _end_rendering(self):
        # We quit pygame display
        pygame.display.quit()

    def _sample_full(self):
        # We sample a value
        coords = np.random.uniform(size=self.nb_params())
        # We generate an image
        output = self.draw(coords.tolist())

        return output

    def _sample_sub(self):
        # We sample a value
        coords = np.random.uniform(.0, .5, self.nb_params())
        # We generate an image
        output = self.draw(coords.tolist())

        return output

    def _sample_corr(self):
        # We sample on a circle
        x = np.random.uniform()
        y = x
        coords = [x,y]
        # We generate an image
        output = self.draw(coords)

        return output

class ArmTwoBallsRenderer(BaseRenderer):
    """
    This class implement a 2 balls renderer

    Factors:
        1: 1st ball x position
        2: 1st ball y position
        3: 2nd ball x position
        4: 2nd ball y position
    """

    def __init__(self, **kwargs):
        # Initialize super
        BaseRenderer.__init__(self, **kwargs)

        # We initialize arguments
        self._factors = np.array([1., 1., 1., 1.])
        self._factors_dims = np.array([1, 1, 1, 1])
        self._params.update({"width": 70,
                             "height": 70,
                             "background_color": "black",
                             "ball_color": "white",
                             "ball_radius": 5})

        # We initialize the rendering environment
        pygame.init()
        self._rendering_vars = {"pygame_screen": None}

    def _init_rendering(self):
        # We generate pygame screen
        self._rendering_vars["pygame_screen"] = pygame.display.set_mode((self._params["width"], self._params["height"]),
                                                                        0, 32)

    def _render(self, fact_coord):
        # We compute real rendering parameters
        x1 = int(fact_coord[0] * self._params["width"])
        y1 = int(fact_coord[1] * self._params["height"])
        x2 = int(fact_coord[2] * self._params["width"])
        y2 = int(fact_coord[3] * self._params["height"])

        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.gfxdraw.aacircle(self._rendering_vars["pygame_screen"],x1,y1,self._params["ball_radius"],pygame.Color(self._params["ball_color"]))
        pygame.gfxdraw.filled_circle(self._rendering_vars["pygame_screen"],x1,y1,self._params["ball_radius"],pygame.Color(self._params["ball_color"]))
        pygame.gfxdraw.aacircle(self._rendering_vars["pygame_screen"],x2,y2,self._params["ball_radius"],pygame.Color(self._params["ball_color"]))
        pygame.gfxdraw.filled_circle(self._rendering_vars["pygame_screen"],x2,y2,self._params["ball_radius"],pygame.Color(self._params["ball_color"]))
        pygame.display.update()

        # We retrieve the image as array
        output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _end_rendering(self):
        # We quit pygame display
        pygame.display.quit()

    def _sample_full(self):
        # We sample a value
        coords = np.random.uniform(size=self.nb_params())
        # We generate an image
        output = self.draw(coords.tolist())

        return output

    def _sample_sub(self):
        # We sample a value
        coords = np.random.uniform(.0, .5, self.nb_params())
        # We generate an image
        output = self.draw(coords.tolist())

        return output

    def _sample_corr(self):
        # We sample with
        x1,y1,x2 = np.random.uniform(size=3)
        y2 = y1
        coords = [x1, y1, x2, y2]
        # We generate an image
        output = self.draw(coords)

        return output

class ArmArrowRenderer(BaseRenderer):
    """
    This class implement a Arrow renderer

    Factors:
        1: arrow x position
        2: arrow y position
        3: arrow rotation
    """

    def __init__(self, **kwargs):
        # Initialize super
        BaseRenderer.__init__(self, **kwargs)

        # We initialize arguments
        self._factors = np.array([1., 1., 1.])
        self._factors_dims = np.array([1, 1, 2])
        self._params.update({"width": 70,
                             "height": 70,
                             "background_color": "black",
                             "arrow_color": "white",
                             "arrow_radius": 10})

        # We initialize the rendering environment
        pygame.init()
        self._rendering_vars = {"pygame_screen": None}

    def _init_rendering(self):
        # We generate pygame screen
        self._rendering_vars["pygame_screen"] = pygame.display.set_mode((self._params["width"], self._params["height"]),
                                                                        0, 32)

    def _render(self, fact_coord):
        # We compute real rendering parameters
        x = int(fact_coord[0] * self._params["width"])
        y = int(fact_coord[1] * self._params["height"])
        r = fact_coord[2] * 2 * np.pi
        p1 = [np.cos(r)*self._params["arrow_radius"]+x, np.sin(r)*self._params["arrow_radius"]+y]
        p2 = [np.cos(r+2*np.pi/3.)*self._params["arrow_radius"]+x, np.sin(r+2*np.pi/3.)*self._params["arrow_radius"]+y]
        p4 = [np.cos(r+4*np.pi/3.)*self._params["arrow_radius"]+x, np.sin(r+4*np.pi/3.)*self._params["arrow_radius"]+y]
        p3 = [x,y]
        p = [p1, p2, p3, p4]

        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.gfxdraw.aapolygon(self._rendering_vars["pygame_screen"],p,pygame.Color(self._params["arrow_color"]))
        pygame.gfxdraw.filled_polygon(self._rendering_vars["pygame_screen"],p,pygame.Color(self._params["arrow_color"]))
        pygame.display.update()

        # We retrieve the image as array
        output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _end_rendering(self):
        # We quit pygame display
        pygame.display.quit()

    def _sample_full(self):
        # We sample a value
        coords = np.random.uniform(size=self.nb_params())
        # We generate an image
        output = self.draw(coords.tolist())

        return output

    def _sample_sub(self):
        # We sample a value
        coords = np.random.uniform(.0, .5, self.nb_params())
        # We generate an image
        output = self.draw(coords.tolist())

        return output

    def _sample_corr(self):
        # We sample with
        x,y = np.random.uniform(size=2)
        r = y
        coords = [x,y,r]
        # We generate an image
        output = self.draw(coords)

        return output

class ArmToolBallRenderer(BaseRenderer):
    """
    This class implements a tool ball renderer.

    Factors:
        1: x position of the ball
        2: y position of the ball
        3: x position of gripable extremity of tool
        4: y position of gripable extremity of tool
        5: orientation of tool
    """

    def __init__(self, **kwargs):
        # Initialize super
        BaseRenderer.__init__(self, **kwargs)

        # We initialize arguments
        self._factors = np.array([1., 1., 1., 1., 1.])
        self._factors_dims = np.array([1, 1, 1, 1, 2])
        self._params.update({"width": 70,
                             "height": 70,
                             "background_color": "black",
                             "ball_radius": 5,
                             "tool_length":20,
                             'grip_radius':2})

        # We initialize the rendering environment
        pygame.init()
        self._rendering_vars = {"pygame_screen": None}

    def _init_rendering(self):
        # We generate pygame screen
        self._rendering_vars["pygame_screen"] = pygame.display.set_mode((self._params["width"], self._params["height"]),
                                                                        0, 32)

    def _render(self, fact_coord):
        # We compute real rendering parameters
        xb = int(fact_coord[0] * self._params["width"])
        yb = int(fact_coord[1] * self._params["height"])
        xt = int(fact_coord[2] * self._params["width"])
        yt = int(fact_coord[3] * self._params["height"])
        rt = 2*np.pi*fact_coord[4]
        xp = int(xt + np.cos(rt) * self._params['tool_length'])
        yp = int(yt + np.sin(rt) * self._params['tool_length'])
        # We draw the image
        self._rendering_vars["pygame_screen"].fill(pygame.Color(self._params["background_color"]))
        pygame.gfxdraw.aacircle(self._rendering_vars["pygame_screen"],xb,yb,self._params['ball_radius'],pygame.Color(self._params["foreground_color"]))
        pygame.gfxdraw.filled_circle(self._rendering_vars["pygame_screen"],xb,yb,self._params['ball_radius'],pygame.Color(self._params["foreground_color"]))
        pygame.gfxdraw.aacircle(self._rendering_vars["pygame_screen"],xt,yt,self._params['grip_radius'],pygame.Color(self._params["foreground_color"]))
        pygame.gfxdraw.filled_circle(self._rendering_vars["pygame_screen"],xt,yt,self._params['grip_radius'],pygame.Color(self._params["foreground_color"]))
        pygame.draw.aaline(self._rendering_vars["pygame_screen"],pygame.Color(self._params['foreground_color']),(xt, yt), (xp, yp))
        pygame.display.update()

        # We retrieve the image as array
        output = pygame.surfarray.array2d(self._rendering_vars["pygame_screen"])

        # We normalize the array
        output = output.astype(np.float32)
        output -= output.min()
        output /= output.max()

        return output

    def _end_rendering(self):
        # We quit pygame display
        pygame.display.quit()

    def _sample_full(self):
        # We sample a value
        coords = np.random.uniform(size=self.nb_params())
        # We generate an image
        output = self.draw(coords.tolist())

        return output

    def _sample_sub(self):
        # We sample a value
        coords = np.random.uniform(.0, .5, self.nb_params())
        # We generate an image
        output = self.draw(coords.tolist())

        return output

    def _sample_corr(self):
        # We sample with
        while True:
            xt,yt,r = np.random.uniform(size=3)
            rt = 2*np.pi*r
            xb = xt + np.cos(rt) * self._params['tool_length']/self._params['width']
            yb = yt + np.sin(rt) * self._params['tool_length']/self._params['height']
            coords = [xb, yb, xt, yt, r]
            if not max(coords)>1. and not min(coords)<0. :
                break
        # We generate an image
        output = self.draw(coords)

        return output
