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

# This module is the heart of the philoyore library; its exact shape will
# come to light as time goes on, but for now we intend that this will be
# the primary user interface to whatever algorithms this library will expose.
# Methods that consume input datasets will usually take them as files; these
# files will probably be tokenized using the philIO (pio) library so that
# this module can process them. The caller may utilize the FilelikeString
# class in the philIO library to convert Python strings (or objects that act
# like Python strings) into file-like objects that will be properly parsed
# by the philIO utilities.

# For now, we are primarily interested in measuring the "distance" between
# two files (where "distance" can mean any number of things depending on the
# algorithm used; in each case, each file is converted into a vector where each
# feature represents the frequency of some word or n-gram). Our general strategy
# for this is as follows:
# 1) Generate the words or n-grams using philIO.
# 2) While streaming in each word/n-gram, keep a count of the number of times
# each word occurs. Do this for both files individually.
# 3) Assign each unique word or n-gram a unique ID number (a non-negative
# integer). In doing so, find the number of unique words/n-grams; call this
# value n.
# 4) Allocate a vector of length n for each input file.
# 5) Populate each element of the vector with the frequencies for the 
# corresponding words/n-grams; then, normalize.
# 6) Compute the distance with the selected distance algorithm.

import philIO as pio
import numpy as np
import scipy, scipy.spatial
import collections

# Given a Counter object (from the collections package), construct a hash
# that maps each unique key to a unique integer ID number. For example, the
# following Counter:
# { 'a' : 7, 'c' : 21, 'b' : 101 }
# might be transformed into the following index hash:
# { 'a' : 0, 'b' : 1, 'c' : 2 }
# Hoewver, the order of the indices is arbitrary, so they may be assigned
# in any order as long as each key as a unique index.
def assign_indices(counter):
    return { k: v for v, k in enumerate(k for k in counter) }

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
# The return value is a 3-tuple, the elements of which are:
# 1) The new vectors mapped to the lower-dimensional space in a list;
# 2) The sum of all the vectors, which can be used for further computations/
#    normalizations;
# 3) A vector of integers that can be used to determine which features were
#    deleted. The vector will be the length of the vectors that were passed
#    into this function; the elements of the vector are the "new" ID's
#    of those vectors, which can be used for reporting. As a special case,
#    if no vectors are deleted, then None will be returned in this place.
# Note that this is not a function for general feature reduction (e.g.
# PCA). That step may come later, after the more low-level feature-
# deletion phase.
def delete_features(features, opts):
    pass

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

# Calculate the distance between two files according to a given distance 
# algorithm and using the given streaming function. The distance algorithm
# must accept two numpy vectors of the same length; whatever it returns
# is directly returned to the caller. The stream function must accept a file
# as an argument and return a stream of features; these objects
# must be hashable and must behave reasonably when compared for equality (the
# normal objects for representing words and n-grams, strings and tuples, satisfy
# these criteria). Usually, you will use the methods in philIO (philIO.words,
# philIO.bigrams, philIO.trigrams, philIO.ngrams(n)) for these purposes. Keep
# in mind that if you want to use the generalized ngrams function, you do need
# to call it because it's curried:
#   distance(f1, f2, streamfn = philIO.ngrams(10))
# However, the other methods (which aren't curried) should just be passed
# directly:
#   distance(f1, f2, streamfn = philIO.words)
def distance(f1, f2, alg = scipy.spatial.distance.cosine, streamfn = pio.words):
    ([v1, v2], ids) = features([streamfn(f1), streamfn(f2)])
    # We don't care about the id's so we can free the space used by them
    del ids
    return alg(v1, v2)
    
