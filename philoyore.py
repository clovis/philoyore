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

# Calculate the distance between two files according to a given distance 
# algorithm and using the given streaming function. The distance algorithm
# must accept two numpy vectors of the same length; whatever it returns
# is directly returned to the caller. The stream function must accept a file
# as an argument and return a stream of some kind of object; these objects
# must be hashable and must behave reasonably when compared for equality (the
# normal objects for generating words and n-grams, strings and tuples, satisfy
# these criteria). Usually, you will use the methods in philIO (philIO.words,
# philIO.bigrams, philIO.trigrams, philIO.ngrams(n)) for these purposes. Keep
# in mind that if you want to use the generalized ngrams function, you do need
# to call it because it's curried:
#   distance(f1, f2, streamfn = philIO.ngrams(10))
# However, the other methods (which aren't curried) should just be passed
# directly:
#   distance(f1, f2, streamfn = philIO.words)
def distance(f1, f2, alg = scipy.spatial.distance.cosine, streamfn = pio.words):
    # Count all terms in input files
    c1 = collections.Counter(streamfn(f1))
    c2 = collections.Counter(streamfn(f2))
    total_count = c1 + c2 
    # Assign unique integer ids
    ids = assign_indices(total_count)
    # Populate feature vectors with counts
    v1 = np.zeros(len(ids), dtype=np.float64)
    v2 = np.zeros(len(ids), dtype=np.float64)
    total_v = np.zeros(len(ids), dtype=np.float64)
    for c, v in [[c1,v1], [c2,v2], [total_count, total_v]]:
        for key in c:
            v[ids[key]] = c[key]
    # Normalize. We do this by dividing each feature by the total count so that
    # v1[n] + v2[n] for any n is 1.0.
    v1 = v1 / total_v
    v2 = v2 / total_v
    # Do a little garbage collection 
    del c1, c2, total_count, ids, total_v
    # Actually compute the distance
    return alg(v1, v2)
    
