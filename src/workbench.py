#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains notebook workbenches to tinker with simulation and embeddings. 

Author: Alexandre Péré

"""

import numpy as np
from bqplot import pyplot as plt
import ipywidgets as ipw
import IPython.display
import simulation
import network

NB_SAMPLES = 100


class SimulationWorkbench(object):
    """
    This class allows to create a workbench to show simulated data
    """

    def __init__(self, simu):
        """
        The initializer of the Simulation Workbench.
        
        Args:
            simu: The simulation object to simulate
        """

        # We check input
        assert issubclass(type(simu), simulation.BaseSimulation)

        # We instantiate arguments
        self._simulation = simu
        self._sliders = [ipw.FloatSlider(min=0., max=1., step=0.01, description=self._simulation.get_factor_labels()[i])
                         for i in range(self._simulation.nb_params())]
        self._factors = list(np.ones(self._simulation.nb_params()))

        # We generate view elements
        self._simu_fig = plt.figure()
        self._simu_fig.background_color =  "red"
        self._simu_plot = plt.heatmap(self._simulation.draw(self._factors, depth=1))
        # We create a slider with callback for each parameter
        for index, slider in enumerate(self._sliders):
            slider.observe(self._callback_closure(index), 'value')
        self._title = ipw.HTML('<h2>Simulation WorkBench</h2>')
        self._caption = ipw.HTML('Manipulate the generative factors:')

        # We layout the different parts
        left_elmts = [self._title, self._caption] + self._sliders
        left_layout = ipw.Layout(padding='50px 0px 0px 0px')
        left_pane = ipw.VBox(left_elmts, layout=left_layout)
        self._simu_fig.layout = ipw.Layout(width='50%')
        self._layout = ipw.HBox([left_pane, self._simu_fig])
        IPython.display.display(self._layout)

    def _callback_closure(self, index):
        """
        The callback closure that allows to create on fly callbacks for sliders.
        
        Args:
            index: the index of slider

        Returns:
            the callback.

        """

        def callback(value):
            self._factors[index] = value.new
            self._simu_plot.color = np.flipud(self._simulation.draw(self._factors))

        return callback


class ReconstructionWorkbench(object):
    """
    This class allows to create a workbench to show reconstruction of input data.
    """

    def __init__(self, simu, net):
        """
        The initializer of the Reconstruction Workbench.

        Args:
            simu: The simulation object to simulate
            net: The network to use to reconstruct data
        """

        # We check input
        assert issubclass(type(simu), simulation.BaseSimulation)
        assert issubclass(type(net), network.BaseNetwork)

        # We instantiate arguments
        self._simulation = simu
        self._network = net
        self._sliders = [ipw.FloatSlider(min=0., max=1., step=0.01, description=self._simulation.get_factor_labels()[i])
                         for i in range(self._simulation.nb_params())]
        self._factors = list(np.ones(self._simulation.nb_params()))

        # We generate view elements
        self._simu_fig = plt.figure()
        self._simu_plot = plt.heatmap(self._simulation.draw(self._factors, depth=1))
        self._recons_fig = plt.figure()
        self._recons_plot = plt.heatmap(self._simulation.draw(self._factors, depth=1))
        # We create a slider with callback for each parameter
        for index, slider in enumerate(self._sliders):
            slider.observe(self._callback_closure(index), 'value')
        self._title = ipw.HTML('<h2>Reconstruction WorkBench</h2>')
        self._caption = ipw.HTML('Manipulate the generative factors:')

        #  We layout the different parts
        left_elmts = [self._title, self._caption] + self._sliders
        left_layout = ipw.Layout(padding='50px 0px 0px 0px')
        left_pane = ipw.VBox(left_elmts, layout=left_layout)
        self._simu_fig.layout = ipw.Layout(width='25%')
        self._recons_fig.layout = ipw.Layout(width='25%')
        self._layout = ipw.HBox([left_pane, self._simu_fig, self._recons_fig])
        IPython.display.display(self._layout)

    def _callback_closure(self, index):
        """
        The callback closure that allows to create on fly callbacks for sliders.

        Args:
            index: the index of slider

        Returns:
            the callback.

        """

        def callback(value):
            in_size = self._network._net_input.shape[-1]
            self._factors[index] = value.new
            image = self._simulation.draw(self._factors)
            self._simu_plot.color = np.flipud(image)
            if not in_size == image.size:
                image = self._simulation.draw(self._factors, depth=3)
                recons = self._network.evaluate_output(image.reshape(1, -1), disable_progress=True).reshape(image.shape).sum(axis=-1)
            else:
                recons = self._network.evaluate_output(image.reshape(1, -1), disable_progress=True).reshape(image.shape)
            self._recons_plot.color = np.flipud(recons)

        return callback


class LatentSpaceWorkbench(object):
    """
    This class allows to create a workbench to observe latent variables while manipulating generative factors.
    """

    def __init__(self, simu, net, emb_index=0):
        """
        The initializer of the Latent Embedding Workbench.

        Args:
            simu: The simulation object to simulate
            net: The NN trained to extract embedding
        """

        # We check input
        assert issubclass(type(simu), simulation.BaseSimulation)
        assert issubclass(type(net), network.BaseNetwork)

        # We instantiate arguments
        self._simulation = simu
        self._emb_index = emb_index
        self._network = net
        self._sliders = [ipw.FloatSlider(min=0., max=1., step=0.01, description=self._simulation.get_factor_labels()[i])
                         for i in range(self._simulation.nb_params())]
        self._factors = list(np.ones(self._simulation.nb_params()))
        self._latent_serie = np.zeros([self._network.get_latent_size()[-1], NB_SAMPLES])

        # We generate view elements
        self._latent_fig = plt.figure()
        self._latent_plot = plt.plot(range(NB_SAMPLES), self._latent_serie)

        # We create a slider with callback for each parameter
        for index, slider in enumerate(self._sliders):
            slider.observe(self._callback_closure(index), 'value')
        self._title = ipw.HTML('<h2>Latent Space WorkBench</h2>')
        self._caption = ipw.HTML('Manipulate the generative factors:')

        #  We layout the different parts
        left_elmts = [self._title, self._caption] + self._sliders
        left_layout = ipw.Layout(padding='50px 0px 0px 0px')
        left_pane = ipw.VBox(left_elmts, layout=left_layout)
        self._latent_fig.layout = ipw.Layout(width='50%')
        self._layout = ipw.HBox([left_pane, self._latent_fig])
        IPython.display.display(self._layout)

    def _callback_closure(self, index):
        """
        The callback closure that allows to create on fly callbacks for sliders.

        Args:
            index: the index of slider

        Returns:
            the callback.

        """

        def callback(value):
            in_size = self._network._net_input.shape[-1]
            self._factors[index] = value.new
            image = self._simulation.draw(self._factors)
            if not in_size == image.size:
                image = self._simulation.draw(self._factors, depth=3)
            latent = self._network.evaluate_latent(image.reshape(1,-1), disable_progress=True).squeeze()[self._emb_index]
            self._latent_serie = np.roll(self._latent_serie,-1,axis=1)
            self._latent_serie[:,-1] = latent
            self._latent_plot.y = self._latent_serie
            
        return callback

