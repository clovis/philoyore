# =======================================================================
#    Copyright (C) 2014  Richard Stewart
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ========================================================================

import collections
import numpy as np
import philoyore.util as putil
import scipy.spatial.distance as dist
import philoyore.dist as pdist
import copy

# A FeatureSet, like a Corpus, is a collection of objects representing the
# frequencies of features in documents; however, the feature set is a sequence
# of "feature vectors", which are vectors representing frequencies of particular
# features. This is good because these are easier to deal with mathematically.
# The object has a number of attributes:
# - id_to_feature: An array mapping feature indices in the feature vectors to
#                  the meaning of those features (i.e. the text or tuple that
#                  that element of the feature vector represents).
# - feature_to_id: A dictionary that performs the opposite operation as the 
#                  above attribute; it maps features to their feature indices.
# - feature_vecs: The list of feature vectors proper. The feature vectors
#                 are just numpy arrays.
# - refs: This is an array that bridges the gap between the FeatureSet and
#         the Corpus the FeatureSet is derived from. The length of this
#         list is the length of the feature_vecs list; if refs[i] = j, that
#         tells you that the feature vector i corresponds to the document j in
#         the Corpus that derived this FeatureSet. (There is no way to
#         remove documents from the FeatureSet for now, so this attribute is
#         useless in the meantime.)
# - total: The sum of all the feature vectors, which is commonly needed 
#          information for a number of operations. This is a cached value:
#          it may be None if the total has not been computed by the library.
# TODO: Look at strategies for reducing the number of docs in a feature
#       set (or maybe that should be done at the Corpus level?)
class FeatureSet:
    # The initializer is a Corpus; command-line arguments are also accepted
    # that can be used to perform certain operations at the time of 
    # instantiation.
    def __init__(self, corpus, **kwargs):
        self.id_to_feature = list(k for k in corpus.total)
        self.feature_to_id = { v : k for k, v in enumerate(self.id_to_feature) }
        self.feature_vecs = np.zeros((len(corpus), len(self.id_to_feature)))
        for i in range(len(self)):
            for feature in corpus[i]:
                index = self.feature_to_id[feature]
                self.feature_vecs[i][index] = corpus[i][feature]
        self.refs = list(range(len(corpus)))
        self.clear_cache()
        self.process(kwargs)

    def __len__(self):
        return len(self.feature_vecs)
    def __getitem__(self, key):
        return self.feature_vecs[key]
    def __iter__(self):
        return iter(self.feature_vecs)

    def clone(self):
        new = copy.copy(self)
        new.feature_vecs = np.copy(new.feature_vecs)
        return new

    # A catch-all method for reducing features in a dataset. The argument is
    # a dictionary of "options" which allows the caller to choose how to
    # reduce the features in the dataset. Possible options are:
    # - minfreq: Minimum proportion of vectors a feature must be in (between 0.0
    #            and 1.0)
    # - maxfreq: Maximum proportion of vectors a feature must be in
    # - minocc: Minimum total "magnitude" of a feature in the entire dataset
    # - maxocc: Maximum total magnitude of a feature in the entire dataset
    # The feature vectors of the feature set will be statefully updated and the
    # id_to_feature and feature_to_id attributes will be updated in order to
    # reflect the changes that were made.
    # TODO: Consider other methods of feature deletion.
    def delete_features(self, opts):
        initial_length = len(self[0])
        indices = range(initial_length)
        # Return immediately if there are no options
        if len(opts) == 0:
            return
        if 'minocc' in opts or 'maxocc' in opts:
            self.find_total()
        if 'minfreq' in opts or 'maxfreq' in opts:
            self.find_props()
        
        if 'minocc' in opts:
            indices = filter(lambda i: self.total[i] >= opts['minocc'], 
                             indices)
        if 'maxocc' in opts:
            indices = filter(lambda i: self.total[i] <= opts['maxocc'],
                             indices)
        if 'minfreq' in opts:
            indices = filter(lambda i: self.props[i] >= opts['minfreq'],
                             indices)
        if 'maxfreq' in opts:
            indices = filter(lambda i: self.propsx[i] <= opts['maxfreq'],
                             indices)
        
        # Return immediately if no features turned out to be deleted
        if len(indices) == initial_length:
            return
        to_delete = set(range(initial_length)) - set(indices)
        # Here, the 1 means we want to delete those columns that are in the
        # list of elements to_delete
        self.feature_vecs = np.delete(self.feature_vecs, list(to_delete), 1)
        self.id_to_feature = list(self.id_to_feature[i] for i in indices)
        self.feature_to_id = { v : k for k, v in enumerate(id_to_feature) }
        self.clear_cache()

    # Statefully normalize the feature vectors according to the given method
    # string. Accepted methods include 'simple, 'tf-idf', 'tf-idf-log2', 
    # 'tf-idf-nolog', and 'none'. 'tf-idf' takes the natural logarithm of
    # its idf values, and 'tf-idf-nolog' doesn't take the log of the idf.
    # 'simple' just divides each of the features in each vector by the
    # sum of each feature, scaling each feature to some number between 0.0
    # and 1.0.
    def normalize(self, method):
        if method == 'simple':
            self.find_total()
            for i in range(len(self)):
                self.feature_vecs[i] /= self.total
            self.clear_cache()
        elif method == 'tf-idf' or method == 'tf-idf-log2' or \
                method == 'tf-idf-nolog':
            self.find_props()
            idf_fun = { 'tf-idf' : np.log, 
                        'tf-idf-log2' : np.log,
                        'tf-idf-nolog' : lambda x: x }[method]
            idf = idf_fun(1.0 / self.props)
            for i in range(len(self)):
                self.feature_vecs[i] *= idf
            self.clear_cache()
        elif method == 'none':
            pass
        else:
            raise RuntimeError, 'Unknown normalization method ' + method 

    # This method allows you to do some high-level operations on feature 
    # sets, including feature deletion and normalization. Currently, you can
    # invoke feature deletion from this method, as well as normalization.
    # As the methods for feature/dataset reduction become more robust, so
    # the interface in this function must change as well.
    def process(self, delete = {}, norm = 'none'):
        self.delete_features(delete)
        self.normalize(norm)

    # Find the distance between the two feature vectors at the two given
    # indices with the given algorithm (which may be a string or a function;
    # if it is a string, it refers to a binary distance function in the 
    # scipy.spatial.distance module). The string strategy is fastest if you
    # can find what you need there!
    def dist(self, i1, i2, alg = 'euclidean'):
        algfn = pdist.distance(alg)
        return algfn(self[i1], self[i2])

    # Find the pairwise distance between all vectors whose indices are in the
    # given list. As above, the alg argument can be a string (referring to
    # a scipy.spatial.distance method) or a function. Pass in None as the
    # list of indices in order to find the pairwise distance between all 
    # vectors. A condensed distance matrix is returned ("condensed distance
    # matrix" meaning the same thing it does in the scipy library). A method
    # in philoyore.util is provided in order to interface with these methods.
    def pdist(self, indices = None, alg = 'euclidean'):
        if indices is None:
            l = len(self)
            ret = dist.pdist(self.feature_vecs, metric = alg)
        else:
            l = len(indices)
            ret = dist.pdist(self[indices], metric = alg)
        return pdist.CondensedDistanceMatrix(l, ret)

    # Consumes two lists of indices and finds the distance between each pair
    # in each collection (as with the scipy.spatial.distance.cdist method).
    # As with the above method, use a string or function for the alg parameter.
    # A distance matrix is returned. 
    def cdist(self, i1, i2, alg = 'euclidean'):
        return dist.cdist(self[i1], self[i2], metric = alg)
    
    # Find the sum of all feature vectors and store it in self.total.
    def find_total(self, force = False):
        if self.total is not None and force is False:
            return
        else:
            self.total = putil.total(self.feature_vecs)

    # Find the "proportions" of all features and store the resultant vector
    # in self.props. self.props[0] will be the proportion of feature vectors the
    # first feature is nonzero in, and so on and so forth.
    def find_props(self, force = False):
        if self.props is not None and force is False:
            return
        else:
            self.props = putil.proportions(self.feature_vecs)

    # If the self.feature_vecs array is statefully updated, call this method to
    # clear the cache (which contains some precomputed useful values.
    def clear_cache(self):
        self.total = None
        self.props = None
