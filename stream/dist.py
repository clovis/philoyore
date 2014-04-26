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

from philoyore.feats.gen import features, normalize
import scipy, scipy.spatial

# Calculate the distance between two streams according to a given distance 
# algorithm. The distance algorithm must accept two numpy vectors of the same 
# length; whatever it returns is directly returned to the caller. 
# This function works by mapping to a feature set, normalizing the two 
# resultant feature vectors, and calculating the result.
def dist(s1, s2, alg = scipy.spatial.distance.cosine):
    (vs, ids) = features([s1, s2])
    # We don't care about the id's so we can free the space used by them
    del ids
    return alg(vs[0], vs[1])
