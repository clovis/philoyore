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

import scipy.spatial.distance as dist

# Represents a single row of a condensed matrix; this is an opaque class.
class CondensedMatrixRow:
    def __init__(self, i, cdm):
        self.i = i
        self.cdm = cdm
    def __getitem__(self, j):
        return self.cdm.get(self.i, j)
    def __len__(self):
        return len(self.cdm)
    def __iter__(self):
        for j in range(len(self.cdm)):
            yield self.cdm.get(self.i, j)

# A condensed matrix that can be accessed in the usual way (cdm[i][j], 
# or cdm.get(i,j)). Can be transformed into a normal square matrix with the
# squareform() method.
class CondensedDistanceMatrix:
    # num_ps is the number of data points in the full distance matrix; 
    # cdm is the condensed distance matrix
    def __init__(self, num_ps, cdm):
        self.n = num_ps
        self.cdm = cdm
    def __getitem__(self, key):
        return CondensedMatrixRow(key, self)
    def __len__(self):
        return self.n
    def __iter__(self):
        for i in range(self.n):
            yield CondensedMatrixRow(i, self)
    def get(self, i, j):
        if i == j:
            return 0.0
        else:
            return self.cdm[(self.n * j) - (j * (j+1) / 2) + i - j - 1]
    def squareform(self):
        return dist.squareform(self.cdm)

# Return the Scipy distance function corresponding to the given string,
# raising a RuntimeError if no such function was found. Returns the input
# parameter if it is not a string.
def distance(s):
    if not isinstance(s, str):
        return s
    h = set(['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
             'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinki',
             'matching', 'rogerstainimoto', 'russelrao', 'sokalsneath',
             'sqeuclidean', 'yule'])
    if s in h:
        return getattr(dist, s)
    else:
        raise RuntimeError, 'No such distance function ' + s + ' found'
