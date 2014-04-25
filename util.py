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

# This is a junk-drawer file for holding some utilities for working with
# feature vectors.

import numpy as np

# Given a list of feature vectors, return their sum.
def total(a):
    return np.array(a).sum(axis=0)

# Given a list of feature vectors, return a vector of proportions of the same
# length as the feature vectors. Each proportion should be the proportion
# of feature vectors that feature is present in. For example, with this 
# dataset:
#   [1, 0,   0]
#   [4, 11,  0]
#   [7, 100, 0]
# Feature 0 is present in all vectors, feature 1 in 2/3rds of the vectors,
# and feature 2 in no vectors, so the proportions would be as follows:
#   [1.0, 0.6666666..., 0.0] 
# "Presence" is determined by a non-zero value in the vector.
def proportions(a):
    res = np.zeros(len(a[0]), dtype=np.float64)
    for i in range(len(a[0])):
        occs = 0
        for j in range(len(a)):
            if a[j][i] != 0:
                occs += 1
        res[i] = float(occs) / float(len(a))
    return res
