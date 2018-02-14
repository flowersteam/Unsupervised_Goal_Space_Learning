#!/usr/bin/python
# coding: utf-8
# Experiments

"""
This module contains experiment class, to help run experiments easily

Author: Alexandre Péré
"""

import uuid
import logging
import os
import h5py
import datetime
import time
import sys
import numpy as np


class BaseExperiment(object):
    """
    This class contain the experiment logic, with logging results saving and all. Made to work with `with` statements.
    """

    def __init__(self, name, params_dict, max_attempts=1, log_path='logs', res_path='results'):
        """
        The initializer of the object

        Args:
            + name: a string used as name of the experiment (may not be unique). Made for the user to quickly recognize.
            + params_dict: a dictionnary containing every parameters of the experiment. Depend on the specific recipe.
            + log_path: The path to the folder to store the logfile
            + res_path: The path to the folder to store the results file (h5py)
        """

        # We initialize the object
        object.__init__(self)
        # We store the paths
        self._log_path = log_path
        self._res_path = res_path
        # We get the identifier of the experiment
        self._uuid = uuid.uuid1().get_hex()
        # We setup the logging
        self._log_lgr = logging.getLogger(self._uuid)
        self._log_lgr.setLevel(logging.DEBUG)
        self._log_fmt = logging.Formatter(fmt="[%(asctime)s] %(levelname)s [Exp: " + self._uuid+"][%(module)s:%(funcName)s:%(lineno)d]  %(message)s")
        self._log_hdl_1 = logging.FileHandler(filename=os.path.join(log_path,self._uuid+'.log'))
        self._log_hdl_1.setLevel(logging.DEBUG)
        self._log_hdl_1.setFormatter(self._log_fmt)
        self._log_hdl_2 = logging.StreamHandler(sys.stdout)
        self._log_hdl_2.setLevel(logging.DEBUG)
        self._log_hdl_2.setFormatter(self._log_fmt)
        self._log_lgr.addHandler(self._log_hdl_1)
        self._log_lgr.addHandler(self._log_hdl_2)
        # We setup the results structure
        self._res_fil = h5py.File(os.path.join(res_path,self._uuid+'.h5'))
        self._res_fil.create_group('parameters')
        self._res_fil.create_group('results')
        self._res_dict = dict()
        # We store the parameters dictionnary
        self._params = params_dict
        for key, item in self._params.items():
            try:
                self._res_fil['parameters'].create_dataset(key, data=item)
            except:
                self._res_fil['parameters'].create_dataset(key, data=str(item))
        # We store the name
        self._name = name
        self._res_fil.create_dataset('name', data=self._name)
        # We store the number of attempts
        self._attempts_ctr = 1
        # We store the date of experiment beginning
        self._exp_beg = datetime.datetime.now()
        self._exp_end = datetime.datetime.now()
        # We store the max number of experiments
        self._max_attempts = max_attempts

    def __enter__(self):
        """
        The entering method, called by the `with` statement.
        """

        # We log informations on the experiment
        self._log_lgr.info("Starting Experiment: %s" % self._name)
        self._log_lgr.info("UUID: %s"%self._uuid)
        self._log_lgr.info("Parameters: %s" % str(self._params))

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        The exit method, called by the `with` statement.
        """

        # We close the logging file
        self._log_hdl_1.close()
        # We flush and close the results
        self._res_fil.flush()
        self._res_fil.close()
        # We log the outcome
        if self._attempts_ctr <= self._max_attempts:
            self._log_lgr.info("Success of Experiment after %i attempts" % self._attempts_ctr)
        else:
            self._log_lgr.info("Failure of Experiment after reaching maximal amounts of attempts")
        # We log the duration
        self._log_lgr.info("Duration: %s " %str(self._exp_end-self._exp_beg))
        self._log_lgr.info("End of Experiment")

    def get_uuid(self):
        """
        This method allows to retrieve the uuid of the experiment.
        """

        return self._uuid

    def run(self):
        """
        The public experiment running method. Manages the multiple attempts logic.
        """

        # We reset the date of start of experiment
        self._exp_beg = datetime.datetime.now()
        # We run the experiment logic
        while True:
            try:
                self._run()
                break
            except Exception as e:
                self._log_lgr.warning('Error occured during attempt %i'%self._attempts_ctr)
                self._log_lgr.error(e, exc_info=True)
                if self._attempts_ctr == self._max_attempts:
                    self._log_lgr.critical("The Experiment was aborted after reaching the maximal number of attempts. Correct the bugs!")
                    self._attempts_ctr += 1
                    break
                else:
                    self._log_lgr.info('Maximal number of attempts is not yet reached. We give it another try.')
                    self._attempts_ctr += 1
                continue
        # We reset the date of experiment End
        self._exp_end = datetime.datetime.now()

        # We save the results
        for key, item in self._res_dict.items():
            try:
                self._res_fil['results'].create_dataset(key, data=item)
            except:
                self._res_fil['results'].create_dataset(key, data=str(item))

        return self._res_dict

    def _run(self):
        """
        The Experiment logic.
        """

        # Initialize Simulation

        # Generate training data using Renderer based on input arguments

        # Train the representation model

        # Perform the GB exploration

        # Compute the measures

        # Flush the results

        self._log_lgr.info("You launched the Virtual Experiment! Nothing Happened!")
        self._res_dict['result'] = np.random.randn(10)
        raise Exception("Dummy Exception")
