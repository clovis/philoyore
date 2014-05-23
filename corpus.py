# =======================================================================
# Copyright (C) 2014 Richard Stewart
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ========================================================================

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer, TfidfTransformer
import sklearn.metrics.pairwise as dist
import sklearn.preprocessing as pre
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy as sp

# A corpus is a collection of vectorized documents. To construct a corpus,
# a collection of "documents" must be passed in; these documents will be 
# tokenized by the Scikit library. The input to the Corpus constructor is
# a collection of *lists* of documents; each list of documents is called 
# a "sub-corpus" and represents a logical grouping within the corpus. These 
# groups can be used to perform some high-level analyses on the sub-corpora 
# later. Sub-corpora can also be specified after the construction of the
# corpus. So the structure of a corpus is like this:
# __________________________________ CORPUS ______________________________
#           /                /                                    \
#          /                /                                      \
#     SUB-CORPUS       SUB-CORPUS                  ......       SUB-CORPUS
# ________________   ________________                          ___________
#   /  /        |      /  |       \                            / |       \
#  DOC DOC ... DOC    DOC DOC ... DOC                        DOC DOC ... DOC
# That is, a corpus is a collection of sub-corpora, and a sub-corpus is a
# collection of documents.
class Corpus:
    # The input `groups` is a Listof(Listof(Document)) (that is, a 
    # Listof(Subcorpus). This corresponds in practice to a
    # Listof(Listof(Filename)). This can be changed by changing the
    # 'input' named parameter to 'file' or 'content' (these correspond
    # to options in the CountVectorizer class in the scikit library).
    # Extra input options will be passed directly to the CountVectorizer
    # or TfidfVectorizer constructor method, depending on which `strategy`
    # is selected ('count' for CountVectorizer or 'tf-idf' for TfidfVectorizer).
    # Use 'hashingcount' as the strategy to use the `HashingVectorizer` from 
    # the Scikit library; use 'hashingtf-idf' as the strategy to use the
    # hashing strategy but to pipe the result into a Tf-Idf transformer.
    # A set of subcorpora will be automatically generated according to the 
    # groups that were passed in; they will automatically be numbered starting
    # from 0.
    def __init__(self, groups, strategy = 'count', input = 'filename', 
                 scale = True, **kwargs):
        all_groups = sum(groups, [])
        # Make our vectorizer
        if strategy == 'count':
            self.vectorizer = CountVectorizer(input = input, dtype = np.float64,
                                              **kwargs)
        elif strategy == 'hashingcount' or strategy == 'hashingtf-idf':
            self.vectorizer = HashingVectorizer(input = input, 
                                                dtype = np.float64, **kwargs)
        elif strategy == 'tf-idf':
            self.vectorizer = TfidfVectorizer(input = input, dtype = np.float64,
                                              **kwargs)
        else:
            raise RuntimeError, 'Unrecognized vectorization strategy ' + \
                strategy
        # Vectorize
        self.vecs = self.vectorizer.fit_transform(all_groups)
        # Transform if we need to
        if strategy == 'hashingtf-idf':
            self.vecs = TfidfTransformer().fit_transform(self.vecs)
        self.vecs = self.vecs.tocsr()
        # Scale if we need to
        self.vecs /= self.vecs.max()
        # Load up our subcorpora
        self.subcorpora_indices = {}
        index = 0
        for i, group in enumerate(groups):
            self.subcorpora_indices[i] = np.array(range(index, 
                                                        index + len(group)))
            index += len(group)
    def get_subcorpus(self, key):
        return self.vecs[self.subcorpora_indices[key]]
    def get_subcorpora(self, keys):
        return self.vecs[reduce(lambda x, y: np.concatenate((x,y)), 
                                [self.subcorpora_indices[key] for key in keys])]
    # The input `group` is a list of 2-tuples of the following form:
    # (SUBCORPUSKEY, INDEX)
    # The input `key` parameter is the key that will be tied to this group, 
    # which can be any object (probably an integer or string). For example,
    # if we want to create a subcorpus consisting of the 0th document from
    # the subcorpora with keys "a", "b", and "c", we might do it with this
    # call:
    # corpus.add_subcorpus('new', [('a', 0), ('b', 0), ('c', 0)])
    def add_subcorpus(self, key, group):
        self.subcorpora_indices[key] = \
            np.array([self.subcorpora_indices[sk][i] for sk, i in group])
    def del_subcorpus(self, key):
        del self.subcorpora_indices[key]
    # Return a list of all the subcorpus keys.
    def subcorpora_list(self):
        return self.subcorpora_indices.keys()
    def features(self):
        return self.vectorizer.get_feature_names()
    def feature_idx(self, feature):
        return self.vectorizer.vocabulary_.get(feature)
    # X_subcorp and Y_subcorp should be subcorpus keys. Returns the distance
    # matrix corresponding to the given parameters; this function merely
    # calls the pairwise_distances function provided by scikit-learn.
    def distance(self, X_subcorp = 0, Y_subcorp = None, n_jobs = -1, **kwargs):
        X = self.get_subcorpus(X_subcorp)
        Y = None if Y_subcorp is None else self.get_subcorpus(Y_subcorp)
        return dist.pairwise_distances(X = X, Y = Y, **kwargs)
    # Returns a KNeighborsClassifier object from the scikit-learn library.
    # The `corpora` parameter should be a list of subcorpora keys that will
    # constitute the labels of the examples.
    def kneighbors(self, subcorpora, **kwargs):
        neigh = KNeighborsClassifier(**kwargs)
        X = self.get_subcorpora(subcorpora)
        y = sum([[k]*len(self.subcorpora_indices[k]) for k in subcorpora], [])
        neigh.fit(X = X, y = y)
        return neigh

# Convert a list of files to a corpus with one subcorpus.
def from_files(fs, **kwargs):
    return Corpus([fs], input = 'file', **kwargs)

# Convert a list of filenames to a corpus with one subcorpus.
def from_filenames(fns, **kwargs):
    return Corpus([fns], **kwargs)

# Convert a list of strings to a corpus with one subcorpus.
def from_strings(ss, **kwargs):
    return Corpus([ss], input = 'content', **kwargs)

# Convert a list of lists of files to a corpus.
def from_file_lists(fss, **kwargs):
    return Corpus(fss, input = 'file', **kwargs)

# Convert a list of lists of fileneames to a corpus (the default behavior of 
# the constructor).
def from_filename_lists(fnss, **kwargs):
    return Corpus(fnss, **kwargs)

# Convert a list of lists of strings to a corpus.
def from_string_lists(sss, *kwargs):
    return Corpus(sss, input = 'content', **kwargs)
