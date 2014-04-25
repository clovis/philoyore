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
# relative frequency of a particular feature; The meaning of that feature
# can be yielded by the second return value of the function, which is a 
# dictionary that maps features to indices (each index being an integer i
# such that 0 <= i < n where n is the number of features). We currently
# construct the feature vectors simply: we count all the features for each
# stream, then normalize by dividing each feature in each vector by the total
# for that feature across all vectors. In other words, for a feature-count x, 
# we have
#   x_normalized = x / sum(x for all feature vectors)
# As an example, consider this set of feature streams:
#   ["hello", "world"]
#   ["goodbye", "world"]
# ... then each of ["hello", "goodbye", "world"] will be assigned a feature
# ID arbitrarily. Presume the ID's are assigned in that order; then the ID
# dictionary that will be returned is
#   { "hello" : 0, "goodbye" : 1, "world" : 2 }
# Now that each feature has a number, we can construct the feature vectors:
#   [1.0, 0.0, 0.5]
#   [0.0, 1.0, 0.5]
# This reflects that the first stream has every single occurrence of "hello"
# and half of the occurrences of "world", and that the second stream has every
# single occurrence of "goodbye" and half of the occurrences of "world". The
# tuple that will be returned is the set of feature vectors and the ID's:
#  ( [(FEATURE1), (FEATURE2)], (IDS) )
# In the future, we will probably have to look to making this method more 
# robust by allowing for especially common/uncommon features to be
# removed or weighted more or less heavily; this will allow users to make
# their calculations more precise or meaningful.
# Note that the ID dict maps features to id's, and not the other way around.
# If you want to quickly find the feature associated with a given ID,
# flip the hash "inside out" (that is, make a hash that maps ID's to 
# features from the output ID hash).
def features(streams):
    def assign_indices(counter):
        return { k: v for v, k in enumerate(k for k in counter) }
    counts = [collections.Counter(s) for s in streams]
    total_counts = sum(counts, collections.Counter())
    ids = assign_indices(total_counts)
    feature_vecs =  [np.zeros(len(ids), dtype=np.float64) for c in counts]
    total_features = np.zeros(len(ids), dtype=np.float64)
    for i in range(len(feature_vecs)):
        for key in counts[i]:
            feature_vecs[i][ids[key]] = counts[i][key]
    for key in total_counts:
        total_features[ids[key]] = total_counts[key]
    for i in range(len(feature_vecs)):
        feature_vecs[i] /= total_features
    return (feature_vecs, ids)

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
