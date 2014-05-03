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

# A FeatureSet, like a StreamSet, is a collection of objects representing the
# frequencies of features in streams; however, the feature set is a sequence
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
# - normalized: A boolean value that is True if the feature set is normalized
#               and False otherwise. This will be set to True if you call the
#               normalize() method; it is conceivable some operations may
#               require normalization before you can use them. If you call
#               the normalize() method when this attribute is True, the 
#               normalization will be skipped. 
# - refs: This is an array that bridges the gap between the FeatureSet and
#         the StreamSet the FeatureSet is derived from. The length of this
#         list is the length of the feature_vecs list; if refs[i] = j, that
#         tells you that the feature vector i corresponds to the stream j in
#         the StreamSet that derived this FeatureSet. (There is no way to
#         remove streams from the FeatureSet for now, so this attribute is
#         useless in the meantime.)
# - total: The sum of all the feature vectors, which is commonly needed 
#          information for a number of operations. This is a cached value:
#          it may be None if the total has not been computed by the library.
# TODO: Look at strategies for reducing the number of streams in a feature
#       set (or maybe that should be done at the StreamSet level?)
class FeatureSet:
    # The initializer is a StreamSet; command-line arguments are also accepted
    # that can be used to perform certain operations at the time of 
    # instantiation.
    def __init__(self, streamset, **kwargs):
        self.id_to_feature = list(k for k in streamset.total)
        self.feature_to_id = { v : k for k, v in enumerate(id_to_feature) }
        self.feature_vecs = np.zeros((len(streamset), len(self.id_to_feature)))
        for i in range(len(self)):
            for feature in streamset[i]:
                self.feature_vecs[i][self.feature_to_id[feature]] = \
                    streamset[i][key]
        self.normalized = False
        self.refs = list(range(len(streamset)))
        self.total = None
        self.process(kwargs)

    def __len__(self):
        return len(self.feature_vecs)
    def __getitem__(self, key):
        return self.feature_vecs[key]
    def __iter__(self):
        return iter(self.feature_vecs)

    # A catch-all method for reducing features in a dataset. The argument is
    # a dictionary of "options" which allows the caller to choose how to
    # reduce the features in the dataset. Possible options are:
    # - minfreq: Minimum proportion of vectors a feature must be in (between 0.0
    #            and 1.0
    # - maxfreq: Maximum proportion of vectors a feature must be in
    # - minocc: Minimum total "magnitude" of a feature in the entire dataset
    # - maxocc: Maximum total magnitude of a feature in the entire dataset
    # The feature vectors of the feature set will be statefully updated and the
    # id_to_feature and feature_to_id attributes will be updated in order to
    # reflect the changes that were made.
    # TODO: Consider other methods of feature deletion.
    def delete_features(self, opts):
        indices = range(len(self[0]))
        if len(opts) == 0:
            return
        if 'minocc' in opts or 'maxocc' in opts:
            self.find_total()
        if 'minfreq' in opts or 'maxfreq' in opts:
            prop_features = putil.proportions(self.feature_vecs)
        
        if 'minocc' in opts:
            indices = filter(lambda i: self.total[i] >= opts['minocc'], 
                             indices)
        if 'maxocc' in opts:
            indices = filter(lambda i: self.total[i] <= opts['maxocc'],
                             indices)
        if 'minfreq' in opts:
            indices = filter(lambda i: prop_features[i] >= opts['minfreq'],
                             indices)
        if 'maxfreq' in opts:
            indices = filter(lambda i: prop_features[i] <= opts['maxfreq'],
                             indices)
        
        to_delete = set(range(len(self[0]))) - set(indices)
        # Here, the 1 means we want to delete those columns that are in the
        # list of elements to_delete
        self.feature_vecs = np.delete(self.feature_vecs, list(to_delete), 1)
        self.id_to_feature = list(self.id_to_feature[i] for i in indices)
        self.feature_to_id = { v : k for k, v in enumerate(id_to_feature) }
        self.total = None

    # Statefully normalize the feature vectors. We currently normalize by 
    # totaling the features and dividing each element by the total for that 
    # element (so the minimum normalized feature value is 0.0 and the maximum 
    # is 1.0 and all instances of a feature sum to 1.0). 
    # TODO For robustness, look at normalization in different ways.
    def normalize(self):
        if self.normalized:
            return
        self.find_total()
        for i in range(len(self)):
            self.feature_vecs[i] /= self.total

    # This method allows you to do some high-level operations on feature 
    # sets, including feature deletion and normalization. Currently, you can
    # invoke feature deletion from this method, as well as normalization.
    # As the methods for feature/dataset reduction become more robust, so
    # the interface in this function must change as well.
    def process(self, delete = {}, norm = True):
        self.delete_features(delete)
        if norm:
            self.normalize()

    # Find the distance between the two feature vectors at the two given
    # indices with the given algorithm (which may be a string or a function;
    # if it is a string, it refers to a binary distance function in the 
    # scipy.spatial.distance module). The string strategy is fastest if you
    # can find what you need there!
    def dist(self, i1, i2, alg = 'euclidean'):
        algfn = putil.distance(alg)
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
