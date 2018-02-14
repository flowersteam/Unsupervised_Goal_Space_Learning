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


    with open(os.path.join(params['path'], 'config.json'), 'w') as f:
        json.dump(params, f, separators=(',\n', ': '))


    logger.info("Instantiating the Embedding")
    if params['environment'] == "armball":
        environment = environments.ArmBallDynamic(dict(m_mins=[-1.] * 7,
                                                       m_maxs=[1.] * 7,
                                                       s_mins=[-1.] * params['nlatents'],
                                                       s_maxs=[1.] * params['nlatents'],
                                                       arm_lengths=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05],
                                                       arm_angle_shift=0.5,
                                                       arm_rest_state=[0.] * 7,
                                                       ball_size=0.1,
                                                       ball_initial_position=[0.6, 0.6]))
        renderer = rendering.ArmBallRenderer(noise=params['noise'],
                                             distractor=params['distractor'],
                                             deformation=params['deformation'],
                                             outliers=params['outliers'])
    elif params['environment'] == 'armarrow':
        environment = environments.ArmArrowDynamic(dict(m_mins=[-1.] * 7,
                                                        m_maxs=[1.] * 7,
                                                        s_mins=[-1.] * params['nlatents'],
                                                        s_maxs=[1.] * params['nlatents'],
                                                        arm_lengths=[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05],
                                                        arm_angle_shift=0.5,
                                                        arm_rest_state=[0.] * 7,
                                                        arrow_size=0.1,
                                                        arrow_initial_pose=[0.6, 0.6, 0.6]))
        renderer = rendering.ArmArrowRenderer(noise=params['noise'],
                                              distractor=params['distractor'],
                                              deformation=params['deformation'],
                                              outliers=params['outliers'])

    logger.info("Instantiating the Embedding Algorithm")
    if params['embedding'] == "ae":
        representation = embeddings.AutoEncoderEmbedding(emb_size=params['nlatents'],
                                                   logs_path=params['path'],
                                                   name=params['name'])
    elif params['embedding'] == "vae":
        representation = embeddings.VariationalAutoEncoderEmbedding(emb_size=params['nlatents'],
                                                              logs_path=params['path'],
                                                              name=params['name'])
    elif params['embedding'] == "rfvae":
        representation = embeddings.RadialVariationalAutoEncoderEmbedding(emb_size=params['nlatents'],
                                                                    logs_path=params['path'],
                                                                    name=params['name'])
    elif params['embedding'] == "isomap":
        representation = embeddings.IsomapEmbedding(emb_size=params['nlatents'],
                                                              logs_path=params['path'],
                                                              name=params['name'])
    elif params['embedding'] == "pca":
        representation = embeddings.PcaEmbedding(emb_size=params['nlatents'],
                                                              logs_path=params['path'],
                                                              name=params['name'])

    logger.info("Sampling the state space")
    X = renderer.sample(nb_samples=params['nb_samples'], type='full')
    np.save(os.path.join(params['path'], 'training_images'), (X*255.).astype(np.uint8))

    logger.info("Training Embedding")
    representation.fit(X)
    representation.save(os.path.join(params['path'], 'representation.pkl'))
    try:
        loss, log_likelihood = representation.get_training_data()
        np.save(os.path.join(params["path"], 'loss'), loss)
        np.save(os.path.join(params["path"], 'lkh'), log_likelihood)
    except:
        pass

    logger.info("Generating Random states")
    samples_states = np.random.uniform(size=[params['nb_samples_manifold'], renderer.nb_params()])
    np.save(os.path.join(params['path'], 'samples_states'), samples_states.astype(np.float16))
    samples_geodesics = utils.manifold_2_space_coordinates(samples_states, renderer.get_factor_dims())
    np.save(os.path.join(params['path'], 'samples_geodesics'), samples_geodesics.astype(np.float16))

    samples_images = np.array([renderer.draw(state) for state in samples_states])
    samples_latents = representation.transform(samples_images)
    np.save(os.path.join(params['path'], 'samples_latents'), samples_latents.astype(np.float16))
    training_latents = representation.transform(X)
    np.save(os.path.join(params['path'], 'training_latents'), training_latents.astype(np.float16))

    if params['embedding'] == 'rfvae':
        samples_latents = representation.transform(samples_images, sampling=True)
        np.save(os.path.join(params['path'], 'samples_latents_rf'), samples_latents.astype(np.float16))
        training_latents = representation.transform(X, sampling=True)
        np.save(os.path.join(params['path'], 'training_latents_rf'), training_latents.astype(np.float16))

    if params['sampling'] == 'kde':
        kde = scipy.stats.gaussian_kde(training_latents.T)

    def transform_method(s):
        s = (s + 1.) / 2.
        image = renderer.draw(s)
        latent = representation.transform(image.reshape([1, -1])).squeeze()
        return latent

    def sampling_method():
        if params['sampling'] == 'uniform':
            sample = np.random.uniform(size=[1, params['nlatents']])
        elif params['sampling'] == 'normal':
            sample = np.random.randn(1, params['nlatents'])
        elif params['sampling'] == 'kde':
            sample = kde.resample(size=1).squeeze()
        return sample.ravel()

    logger.info("Instantiating the explorator")
    explorator = exploration.GoalBabblingExploration(environment=environment,
                                                     explo_ratio=params['explo_ratio'],
                                                     emb_size=params['nlatents'],
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

    parser = argparse.ArgumentParser(prog="Random Goal Babbling on Learned Goal Spaces",
                                     usage="Input the embedding to use and the environment",
                                     description='This script performs experiment on Unsupervised Goal Learning')

    parser.add_argument('embedding', help="the Embedding you want to use", type=str, choices=["pca", "ae", "vae", "rfvae", "isomap"])
    parser.add_argument('environment', help="the Environment you want to use", type=str, choices=["armball", "armarrow"])

    parser.add_argument('--sampling', help='What distribution to use to sample goals', type=str, choices=['kde', 'normal', 'uniform'], default='kde')
    parser.add_argument('--nlatents', help="Number of latent dimensions to use", type=int, default=10)
    parser.add_argument('--noise', help="Level of noise added to the generated images", type=float, default=.0)
    parser.add_argument('--distractor', help="Whether to add a random distractor in the images", type=bool, default=False)
    parser.add_argument('--deformation', help="Level of deformation for the images", type=float, default=0.)
    parser.add_argument('--outliers', help="Level of outliers present in images", type=float, default=0.)
    parser.add_argument('-t', '--test', help='Whether to make a (shorter) test run', action="store_true")
    parser.add_argument('--path', help='Path to the results folder', type=str, default='.')
    parser.add_argument('--name', help='Name of the experiment', type=str, default='')
    parser.add_argument('--verbose', help="Output logs to stream", type=bool, default=False)

    args = vars(parser.parse_args())

    assert os.path.isdir(args['path']), "You provided a wrong path."
    assert args['noise'] <= 1., "Noise must be inferior to 1."
    assert args['deformation'] <= 1., "Deformation must be inferior to 1."
    assert args['outliers'] <= 1., "Outliers must be inferior to 1."

    if args['name'] == '':
        args['name'] = ("RGE-REP %s %s %s"%(args['embedding'], args['environment'], str(datetime.datetime.now()))).title()

    args['path'] = os.path.join(args['path'], args['name'])
    logger = logging.getLogger(args['name'],)
    logger.setLevel(logging.INFO)

    if args['verbose']:
        logger.addHandler(logging.StreamHandler())

    os.mkdir(args['path'])
    handler = logging.FileHandler(os.path.join(args['path'], 'logs.txt'))
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s[%(module)s:%(funcName)s:%(lineno)d]  %(message)s"))
    logger.addHandler(handler)

    exc_code = 0
    try:
        run_experiment(args)
    except:
        logger.exception("Exception occured during experiment")
        exc_code = 999
    finally:
        sys.exit(exc_code)


