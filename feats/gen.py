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

# Generate vectors corresponding to each feature stream in the input list. 
# Each element of each vector is a floating-point value representing the
# frequency of a particular feature; The meaning of that feature
# can be yielded by the second return value of the function, which is a 
# dictionary that maps features to indices (each index being an integer i
# such that 0 <= i < n where n is the number of features). Note that the ID 
# dict maps features to id's, and not the other way around. If you want to 
# quickly find the feature associated with a given ID, flip the hash 
# "inside out" (that is, make a hash that maps ID's to features from the 
# output ID hash).
def raw_features(streams):
    def assign_indices(counter):
        return { k: v for v, k in enumerate(k for k in counter) }
    counts = [collections.Counter(s) for s in streams]
    # TODO We find the indices by adding all the counters together; this may
    # not be as efficient an operation as possible. Have a look at the 
    # efficiency of this and tighten up if necessary.
    ids = assign_indices(sum(counts, collections.Counter()))
    feature_vecs =  [np.zeros(len(ids), dtype=np.float64) for c in counts]
    for i in range(len(feature_vecs)):
        for key in counts[i]:
            feature_vecs[i][ids[key]] = counts[i][key]
    return (feature_vecs, ids)

# Given a set of feature vectors, statefully normalize the vectors. We 
# currently normalize by totaling the features and dividing each
# element by the total for that element (so the minimum normalized feature 
# value is 0.0 and the maximum is 1.0 and all instances of a feature sum to
# 1.0). 
# TODO For robustness, look at normalization in different ways. 
def normalize(features):
    total_features = putil.total(features)
    for i in range(len(features)):
        features[i] /= total_features

# This is a catch-all function for reducing features in a dataset. Two
# arguments are accepted: 1) a list of the feature arrays, and 2) a 
# dictionary of "options" which allows the caller to choose how to 
# reduce the features in the dataset. Possible options are:
# - minfreq: Minimum proportion of vectors a feature must be in;
# - maxfreq: Maximum proportion of vectors a feature must be in;
# - minocc: Minimum number of occurrences a feature must have in the
#           entire dataset in order to be retained;
# - maxocc: Maximum number of occurrences a feature must have in the entire 
#           dataset in order to be retained.
# More options may be added later. 
# The return value is a 2-tuple, the elements of which are:
# 1) The new vectors mapped to the lower-dimensional space in a list;
# 2) A vector of integers that can be used to determine which features were
#    deleted. The vector will be the length of the feature-deleted feature 
#    vectors returned by this function; each element is a pointer to the
#    index that feature had in the non-feature-deleted version of the feature
#    vector. For example, if you pass in a set of feature vectors of length 5,
#    and this function deletes every feature but the 2nd, then this return
#    value would be [1]; that is, the feature vectors that will be returned will
#    be of length 1, and that element is the 2nd element of the original 
#    vectors. 
# Note that this is not a function for general feature reduction (e.g.
# PCA). That step may come later, after the more low-level feature-
# deletion phase.
def delete_features(features, opts):
    # Some setup
    indices = range(len(features[0]))
    if len(opts) == 0:
        return (features, indices)
    if 'minocc' in opts or 'maxocc' in opts:
        sum_features = putil.total(features)
    if 'minfreq' in opts or 'maxfreq' in opts:
        proportion_features = putil.proportions(features)

    # Filter out the unwanted features
    if 'minocc' in opts:
        indices = filter(lambda i: sum_features[i] >= opts['minocc'], indices)
    if 'maxocc' in opts:
        indices = filter(lambda i: sum_features[i] <= opts['maxocc'], indices)
    if 'minfreq' in opts:
        indices = filter(lambda i: proportion_features[i] >= opts['minfreq'],
                         indices)
    if 'maxfreq' in opts:
        indices = filter(lambda i: proportion_features[i] <= opts['maxfreq'],
                         indices)
    
    # Now, indices is an ordered collection of the features we want to keep.
    # We'll actually perform the transformation here.
    return (map(lambda v: v[indices], features), indices)

# This method is a high-level method for converting streams into feature 
# vectors. We call raw_features to get the simple feature vectors, call
# delete_features (if a delete-option dictionary is supplied), and normalize
# (unless we're asked not to normalize by the caller). This function has a 
# high-level declarative bent; callers should generally use this function
# to convert streams to feature vectors, rather than the lower-level (more
# error-prone) methods.
# TODO Look at other strategies for feature-deletion and normalizing and 
# implement them in this function. Consider other parameters that may need
# to be considered in this function (e.g. PCA?)
def features(streams, delete = {}, norm = True):
    features, ids = raw_features(streams)
    features, id_map = delete_features(features, delete)
    if norm:
        normalize(features)
    new_ids = {}
    for key, value in ids.items():
        try:
            index = id_map.index(value)
            new_ids[key] = index
        except ValueError:
            continue
    return (features, new_ids)
