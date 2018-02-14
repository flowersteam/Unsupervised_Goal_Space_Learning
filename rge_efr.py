#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import datetime
import json
import sys
sys.path.append("src")
import environments
import rendering
import embeddings
import utils
import exploration
import measures
import numpy as np
import scipy.stats
import pickle


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s[%(module)s:%(funcName)s:%(lineno)d]  %(message)s")
os.environ["JOBLIB_TEMP_FOLDER"] = "."

def run_experiment(params):

    logger = logging.getLogger(params['name'])

    logger.info("Instantiating the Environment")
    if params['test']:
        params['nb_samples'] = int(1e2)
        params['nb_samples_manifold']  = int(1e2)
        params['nb_samples_divergence']  = int(1e2)
        params['nb_samples_mse']  = int(1e2)
        params['nb_bins_exploration_ratio']  = 10
        params['nb_exploration_iterations']  = int(1e2)
        params['nb_period_callback']  = int(1e2 - 1)
        params['explo_ratio'] = 0.05
    else:
        params['nb_samples'] = int(1e4)
        params['nb_samples_manifold'] =int(1e3)
        params['nb_samples_divergence'] =int(1e3)
        params['nb_samples_mse'] =int(1e2)
        params['nb_bins_exploration_ratio'] =10
        params['nb_exploration_iterations'] =int(5e3)
        params['nb_period_callback'] =int(1e1)
        params['explo_ratio'] = 0.05


    logger.info("Instantiating the Embedding")
    if params['environment'] == "armball":

        params['nstate'] = 2
        params['norder'] = [1,1]

        environment = environments.ArmBallDynamic(dict(m_mins=[-1.] * 7,
                                                       m_maxs=[1.] * 7,
                                                       s_mins=[-1.] * params['nstate'],
                                                       s_maxs=[1.] * params['nstate'],
                                                       arm_lengths=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05],
                                                       arm_angle_shift=0.5,
                                                       arm_rest_state=[0.] * 7,
                                                       ball_size=0.1,
                                                       ball_initial_position=[0.6, 0.6]))

    elif params['environment'] == 'armarrow':

        params['nstate'] = 3
        params['norder'] = [1,1,2]

        environment = environments.ArmArrowDynamic(dict(m_mins=[-1.] * 7,
                                                        m_maxs=[1.] * 7,
                                                        s_mins=[-1.] * params['nstate'],
                                                        s_maxs=[1.] * params['nstate'],
                                                        arm_lengths=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05],
                                                        arm_angle_shift=0.5,
                                                        arm_rest_state=[0.] * 7,
                                                        arrow_size=0.1,
                                                        arrow_initial_pose=[0.6, 0.6, 0.6]))

    with open(os.path.join(params['path'], 'config.json'), 'w') as f:
        json.dump(params, f, separators=(',\n', ': '))


    logger.info("Generating Random states")
    samples_states = np.random.uniform(size=[params['nb_samples_manifold'], params['nstate']])
    np.save(os.path.join(params['path'], 'samples_states'), samples_states.astype(np.float16))
    samples_geodesics = utils.manifold_2_space_coordinates(samples_states, params['norder'])
    np.save(os.path.join(params['path'], 'samples_geodesics'), samples_geodesics.astype(np.float16))

    def transform_method(s):
        latent = (s + 1.) / 2.
        return latent

    def sampling_method():
        sample = np.random.uniform(size=[1, params['nstate']])
        return sample.ravel()

    logger.info("Instantiating the explorator")
    explorator = exploration.GoalBabblingExploration(environment=environment,
                                                     explo_ratio=params['explo_ratio'],
                                                     emb_size=params['nstate'],
                                                     transform_method=transform_method,
                                                     sampling_method=sampling_method,
                                                     callback_period=params['nb_period_callback'])
    explored_states_history = []
    explored_latents_history = []

    def callback_method():
        logger.info("Executing Callback")
        explored_states_history.append(np.array(explorator.get_reached_states()).astype(np.float16))
        explored_latents_history.append(np.array(explorator.get_reached_latents()).astype(np.float16))

    explorator.set_callback(callback_method=callback_method)

    logger.info("Executing Exploration")
    explorator.explore(nb_iterations=params['nb_exploration_iterations'])

    logger.info("Saving Exploration data")
    with open(os.path.join(params['path'], 'explored_states_history.pkl'), 'w') as f:
        pickle.dump(explored_states_history, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(params['path'], 'explored_latents_history.pkl'), 'w') as f:
        pickle.dump(explored_latents_history, f, pickle.HIGHEST_PROTOCOL)

    logger.info("End of experiment reached")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog="Random Goal Exploration on Engineer Space",
                                     usage="Input the environment to use",
                                     description="This script performs experiments on random goal exploration")

    parser.add_argument('environment', help="the Environment you want to use", type=str, choices=["armball", "armarrow"])

    parser.add_argument('-t', '--test', help='Whether to make a (shorter) test run', action="store_true")
    parser.add_argument('--path', help='Path to the results folder', type=str, default='.')
    parser.add_argument('--name', help='Name of the experiment', type=str, default='')
    parser.add_argument('--verbose', help="Output logs to stream", type=bool, default=False)
    args = vars(parser.parse_args())

    assert os.path.isdir(args['path']), "You provided a wrong path."

    if args['name'] == '':
        args['name'] = ("RGE-EFR %s %s"%(args['environment'], str(datetime.datetime.now()))).title()

    args['path'] = os.path.join(args['path'], args['name'])
    logger = logging.getLogger(args['name'],)
    logger.setLevel(logging.INFO)

    if args['verbose']:
        logger.addHandler(logging.StreamHandler())

    os.mkdir(args['path'])
    handler = logging.FileHandler(os.path.join(args['path'], 'logs.txt'))
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s[%(module)s:%(funcName)s:%(lineno)d]  %(message)s"))
    logger.addHandler(handler)

    try:
        run_experiment(args)
    except:
        logger.exception("Exception occured during experiment")
        raise
