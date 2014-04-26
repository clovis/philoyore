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

from philoyore.feats.gen import features
import scipy, scipy.spatial

# Calculate the distance between two streams according to a given distance 
# algorithm. The distance algorithm must accept two numpy vectors of the same 
# length; whatever it returns is directly returned to the caller. 
# This function works by mapping to a feature set, normalizing the two 
# resultant feature vectors, and calculating the result.
# TODO: the "alg" parameter should be a string pointing to a Scipy function
#       if not a distance function
def dist(s1, s2, alg = scipy.spatial.distance.cosine):
    (vs, ids) = features([s1, s2])
    # We don't care about the id's so we can free the space used by them
    del ids
    return alg(vs[0], vs[1])

# TODO: pairwise distance (scipy.spatial.distance.pdist); this should return
#       a "compressed distance matrix" like the pdist method does, though we
#       should also expose an accessor method which converts (i, j) pairwise 
#       indices to indices in the compressed distance matrix, which scipy
#       does not seem to expose for some stupid reason.
# TODO: Coupled distance: take two arrays of streams and compute the distance
#       between each pair using scipy.spatial.distance.cdist
# TODO: Come up with a good design for dealing with the differences in 
#       semantics between "streams" and feature vectors. At a high level, having
#       functions accept lists of streams is a good thing, since we can abstract
#       away the tricky business of converting those streams to feature vectors.
#       However, letting the programmer control this would also be a good idea,
#       since he could, for example, find the features corresponding to a stream
#       set ONCE and then pass that dataset to multiple functions for analysis,
#       i.e., repeating the computation of the feature counts (a non-trivial
#       process) shouldn't be necessary. The project is currently structured
#       in a way that might make that possible; for example, we can have
#       philoyore.stream.dist and philoyore.feats.dist (the former could
#       hook into the latter). Or maybe separating these concerns isn't the
#       best idea and we should only provide operations over feature vectors.
# TODO: Supervised and unsupervised learning tasks once I've straightened out
#       distance.
