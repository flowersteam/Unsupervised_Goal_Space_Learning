#!/usr/bin/python
# coding: utf-8
# Embeddings

"""
This module contains embeddings. We mimic the sklearn interface.

Author: Alexandre Péré
"""

import sklearn.manifold
import sklearn.decomposition
import architectures
import numpy as np
from sklearn.externals import joblib


class BaseEmbedding(object):
    """
    This class contain the embedding logic.
    """

    def __init__(self, emb_size, logs_path="logs", name="None"):
        """
        The initializer of the object. All the parameters of the training should be put in this one.

        Args:
            + emb_size: the size of the embedding to consider.
            + logs_path: the path to logs if needed
            + name: the name if needed for logs
        """

        # We instantiate object
        object.__init__(self)
        # We store the variables
        self._emb_size = emb_size
        self._logs_path = logs_path
        self._name = name

    def fit(self, X):
        """
        This method allows to train an embedding with some data X.
        """

        raise NotImplementedError("Calling a Virtual Method")

    def transform(self, X, sampling=False):
        """
        This method allows to embedd X in embedded space. Depending sampling allows to select the sampling or manifold
        for flow vae.
        """

        raise NotImplementedError("Calling a Virtual Method")

    def get_training_data(self):
        """
        This method allows to retrieve the learning curves, when there are.
        """

        return None, None

    def save(self, path):
        """
        This method saves the network
        """

        raise NotImplementedError("Calling a virtual method")


class IsomapEmbedding(BaseEmbedding):
    """
    The isomap embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = sklearn.manifold.Isomap(n_components=self._emb_size, n_jobs=8)

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.fit(X)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.transform(X)

        return output

    def save(self, path):
        """
        This method saves the network
        """

        joblib.dump(self._model, path)


class AutoEncoderEmbedding(BaseEmbedding):
    """
    The autoencoder embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = architectures.DenseAE(logs_name=self._name+"-AE", path_to_logs=self._logs_path,
                                            lr=1e-3, emb_size=self._emb_size)

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.train(X_train=X, y_train=X, iterations=int(2e5), batch_size=100, disable_progress=True)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.evaluate_latent(X=X, disable_progress=True)[:,0,:]

        return output

    def get_training_data(self):

        # We retrieve the summaries
        loss = self._model.get_summaries(name="Summaries/Scalars/Loss").squeeze()
        likelihood = self._model.get_summaries(name="Summaries/Scalars/LogLikelihood").squeeze()

        return loss, likelihood

    def save(self, path):
        """
        This method saves the network
        """

        self._model.save(path)


class DenoisingAutoEncoderEmbedding(BaseEmbedding):
    """
    The stacked denoising autoencoder embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = architectures.DenseSDAE(logs_name=self._name+"-SDAE", path_to_logs=self._logs_path,
                                              lr=1e-3, emb_size=self._emb_size)

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.train(X_train=X, y_train=X, iterations=int(1e5), batch_size=100, disable_progress=True)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.evaluate_latent(X=X, disable_progress=True)[:,0,:]

        return output

    def get_training_data(self):

        # We retrieve the summaries
        loss = self._model.get_summaries(name="Summaries/Scalars/Loss").squeeze()
        likelihood = self._model.get_summaries(name="Summaries/Scalars/LogLikelihood").squeeze()

        return loss, likelihood

    def save(self, path):
        """
        This method saves the network
        """

        self._model.save(path)


class VariationalAutoEncoderEmbedding(BaseEmbedding):
    """
    The variational autoencoder embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = architectures.DenseVAE(logs_name=self._name+"-VAE", path_to_logs=self._logs_path,
                                            lr=1e-3, emb_size=self._emb_size)

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.train(X_train=X, y_train=X, iterations=int(1e5), batch_size=100, disable_progress=True)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.evaluate_latent(X=X, disable_progress=True)[:,0,:]

        return output

    def get_training_data(self):

        # We retrieve the summaries
        loss = self._model.get_summaries(name="Summaries/Scalars/Loss").squeeze()
        likelihood = self._model.get_summaries(name="Summaries/Scalars/LogLikelihood").squeeze()

        return loss, likelihood

    def save(self, path):
        """
        This method saves the network
        """

        self._model.save(path)


class BetaVariationalAutoEncoderEmbedding(BaseEmbedding):
    """
    The beta variational autoencoder embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = architectures.DenseVAE(logs_name=self._name+"-BVAE", path_to_logs=self._logs_path,
                                            lr=1e-3, emb_size=self._emb_size, beta=4.)

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.train(X_train=X, y_train=X, iterations=int(1e5), batch_size=100, disable_progress=True)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.evaluate_latent(X=X, disable_progress=True)[:,0,:]

        return output

    def get_training_data(self):

        # We retrieve the summaries
        loss = self._model.get_summaries(name="Summaries/Scalars/Loss").squeeze()
        likelihood = self._model.get_summaries(name="Summaries/Scalars/LogLikelihood").squeeze()

        return loss, likelihood

    def save(self, path):
        """
        This method saves the network
        """

        self._model.save(path)


class PlanarVariationalAutoEncoderEmbedding(BaseEmbedding):
    """
    The planar flow variationa autoencoder embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = architectures.DensePlanarVAE(logs_name=self._name+"-PVAE", path_to_logs=self._logs_path,
                                                   lr=1e-3, emb_size=self._emb_size)

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.train(X_train=X, y_train=X, iterations=int(5e4), batch_size=100, disable_progress=True)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.evaluate_latent(X=X, disable_progress=True)
        output = output[:,0,:] if sampling else output[:,1,:]

        return output

    def get_training_data(self):

        # We retrieve the summaries
        loss = self._model.get_summaries(name="Summaries/Scalars/Loss").squeeze()
        likelihood = self._model.get_summaries(name="Summaries/Scalars/LogLikelihood").squeeze()

        return loss, likelihood

    def save(self, path):
        """
        This method saves the network
        """

        self._model.save(path)


class RadialVariationalAutoEncoderEmbedding(BaseEmbedding):
    """
    The radial flow variational autoencoder embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = architectures.DenseRadialVAE(logs_name=self._name+"-RVAE", path_to_logs=self._logs_path,
                                                  lr=1e-3, emb_size=self._emb_size)

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.train(X_train=X, y_train=X, iterations=int(5e4), batch_size=100, disable_progress=True)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.evaluate_latent(X=X, disable_progress=True)
        output = output[:,0,:] if sampling else output[:,1,:]

        return output

    def get_training_data(self):

        # We retrieve the summaries
        loss = self._model.get_summaries(name="Summaries/Scalars/Loss").squeeze()
        likelihood = self._model.get_summaries(name="Summaries/Scalars/LogLikelihood").squeeze()

        return loss, likelihood

    def save(self, path):
        """
        This method saves the network
        """

        self._model.save(path)


class TsneEmbedding(BaseEmbedding):
    """
    The tsne embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = sklearn.manifold.TSNE(n_components=self._emb_size, init='pca', method='exact')

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.fit(X)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.transform(X)

        return output

    def save(self, path):
        """
        This method saves the network
        """

        joblib.dump(self._model, path)


class PcaEmbedding(BaseEmbedding):
    """
    The PCA embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = sklearn.decomposition.TruncatedSVD(n_components=self._emb_size)

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.fit(X)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.transform(X)

        return output

    def save(self, path):
        """
        This method saves the network
        """

        joblib.dump(self._model, path)


class MdsEmbedding(BaseEmbedding):
    """
    The Multi Dimensional Analysis embedding
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)
        # We instantiate the sklearn model
        self._model = sklearn.manifold.MDS(n_components=self._emb_size, n_jobs=8)

    def fit(self, X):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn function
        self._model.fit(X)

    def transform(self, X, sampling=False):

        # We resize the data to be 2dims
        X = X.reshape([X.shape[0],-1])

        # We call the sklearn method
        output = self._model.transform(X)

        return output

    def save(self, path):
        """
        This method saves the network
        """

        joblib.dump(self._model, path)


class PixelEmbedding(BaseEmbedding):
    """
    The pixel embedding, which embeds nothing.
    """

    def __init__(self, *args, **kwargs):

        # We instantiate object
        BaseEmbedding.__init__(self, *args, **kwargs)

    def fit(self, X):

        pass

    def transform(self, X, sampling=False):

        # We resize the image
        X = X.reshape([X.shape[0],-1])

        return X
