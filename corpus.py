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
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, SpectralClustering, Ward, DBSCAN
from sklearn.decomposition import TruncatedSVD
import numpy as np
import scipy as sp
import sys

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
    # ==========================================
    # =========== PREPROCESSING DATA ===========
    # ==========================================
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
                 **kwargs):
        all_groups = sum(groups, [])
        # ======= VECTORIZING =======
        # We vectorize by using the vectorizer utility classes provided in 
        # Scikit. We use all the default keyword arguments for these vectorizers
        # by default though the extra keyword arguments from this method will
        # be passed directly to these constructors.
        if strategy == 'count':
            self.feature_names_available = True
            self.vectorizer = CountVectorizer(input = input, dtype = np.float64,
                                              **kwargs)
        elif strategy == 'hashingcount' or strategy == 'hashingtf-idf':
            self.feature_names_available = False
            self.feature_names_unavailable_reason = "hashing vectorization"
            self.vectorizer = HashingVectorizer(input = input, 
                                                dtype = np.float64, **kwargs)
        elif strategy == 'tf-idf':
            self.feature_names_available = True
            self.vectorizer = TfidfVectorizer(input = input, dtype = np.float64,
                                              **kwargs)
        else:
            raise RuntimeError, 'Unrecognized vectorization strategy ' + \
                strategy
        self.vecs = self.vectorizer.fit_transform(all_groups)
        # Tfidf transform if we need to
        if strategy == 'hashingtf-idf':
            self.vecs = TfidfTransformer().fit_transform(self.vecs)
        # Convert to CSR if not in CSR already
        if self.vecs.format != 'csr':
            self.vecs = self.vecs.tocsr()
        self.sparse = True
        # ======= SUBCORPORA CONSTRUCTION =======
        # Besides vectorizing and providing light wrappers around computational
        # methods from other scientific libraries, this class takes care of
        # subcorpora management for us. These lines of code initialize the
        # initial set of subcorpora, inferred from the input parameters.
        self.subcorpora_indices = {}
        index = 0
        for i, group in enumerate(groups):
            self.subcorpora_indices[i] = np.array(range(index, 
                                                        index + len(group)))
            index += len(group)
    def scale(self):
        # Scaling is an important part of this process: many of our algorithms
        # require our data to be scaled or otherwise standardized. We 
        # do this by scaling features to values between [0,1]. This preserves
        # zero entries in our sparse matrix which is always a desirable 
        # quality when working with this sort of data.
        # Scaling is sort of a convoluted process because Scipy/Scikit
        # doesn't offer a way to do this natively. We transpose the matrix, 
        # convert it to LIL format (which isn't inefficient in this operation),
        # and divide each row (column in the original matrix) by the row's
        # sum before transposing and converting back to CSR. 
        # However, if the matrix is not sparse, we don't have to worry about
        # this and can simply use one of Scikit's utility methods.
        # TODO: Maybe look at profiling to ensure that this strategy really
        # is the least expensive one.
        if self.sparse:
            self.vecs = self.vecs.tolil()
            self.vecs = self.vecs.transpose()
            num_features, _ = self.vecs.shape
            for i in range(num_features):
                self.vecs[i] /= self.vecs[i].sum()
            self.vecs = self.vecs.transpose()
            self.vecs = self.vecs.tocsr()
        else:
            mms = MinMaxScaler(copy = False)
            self.vecs = mms.fit_transform(self.vecs)
            
    # ===============================================
    # =========== WORKING WITH SUBCORPORA ===========
    # ===============================================
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
        if self.feature_names_available:
            return self.vectorizer.get_feature_names()
        else:
            raise RuntimeError, "Features not available due to " + \
                self.feature_names_unavailable_reason
    def feature_idx(self, feature):
        if self.feature_names_available:
            return self.vectorizer.vocabulary_.get(feature)
        else:
            raise RuntimeError, "Features not available due to " + \
                self.feature_names_unavailable_reason
    # ==================================
    # =========== ALGORITHMS ===========
    # ==================================

    # ======= PIPELINING =======
    # Often, we find that we want to perform a series of transformations and
    # computations on a corpus. This function will enable that for us. Call
    # this method with a list of "commands" as its argument, where a command is:
    # - either a method name (e.g. 'distance' or 'kneighbors'); or 
    # - a 2-tuple with a method name (e.g. 'distance' or 'kneighbors')
    #   and a list of arguments (e.g. [0, 1]); or
    # - a 3-tuple with a method name, a list of arguments, and a dictionary
    #   of keyword arguments (e.g. { n_jobs : 10 }). 
    # This function will run all of the commands in sequence, returning
    # all the return values as a list. So an example usage of this might be
    # corpus.pipeline([ 
    #   ('LSA', [], { n_components : 75 }),
    #   ('distance', ['a', 'b'], { n_jobs : 10 }),
    #   ('decision_tree', ['a', 'b', 'c' ])
    # ])
    # The function does NOT run commands in parallel (though this will probably
    # come next); i.e., the list of commands will be run and returned
    # sequentially.
    # The return value is a 2-tuple: a list of return values, or a string
    # describing what went wrong, if anything. None will be in the 2nd spot
    # of the tuple if nothing went wrong.
    def pipeline(self, commands):
        return_values = []
        for command in commands:
            try:
                if isinstance(command, str):
                    return_values.append(getattr(self, command))
                elif len(command) == 2:
                    return_values.append(getattr(self, command[0]), *command[1])
                elif len(command) == 3:
                    return_values.append(getattr(self, command[0]), *command[1],
                                         **command[2])
                else:
                    return (return_values, 
                            "Received invalid command " + str(command))
            except:
                return (return_values, 
                        "Received exception " + sys.exc_info()[1])
        return return_values
                    
                
    # ======= DISTANCE =======
    # X_subcorp and Y_subcorp should be subcorpus keys. Returns the distance
    # matrix corresponding to the given parameters; this function merely
    # calls the pairwise_distances function provided by scikit-learn.
    def distance(self, X_subcorp = 0, Y_subcorp = None, n_jobs = -1, **kwargs):
        X = self.get_subcorpus(X_subcorp)
        Y = None if Y_subcorp is None else self.get_subcorpus(Y_subcorp)
        return pairwise_distances(X = X, Y = Y, **kwargs)
    # ======= CLASSIFICATION =======
    # Classify the text using the given classifier function to generate
    # the classifier. This is a utility method used by the actual user-facing
    # classifier functions and is not meant to be called by the user directly.
    def classify(self, subcorpora, classifier_fn, **kwargs):
        classifier = classifier_fn(**kwargs)
        X = self.get_subcorpora(subcorpora)
        y = sum([[k]*len(self.subcorpora_indices[k]) for k in subcorpora], [])
        classifier.fit(X = X, y = y)
        return classifier
    # Returns a KNeighborsClassifier object from the scikit-learn library.
    # The `corpora` parameter should be a list of subcorpora keys that will
    # constitute the labels of the examples.
    def kneighbors(self, subcorpora, **kwargs):
        return self.classify(subcorpora, KNeighborsClassifier, **kwargs)
    # Returns an SVM classifier.
    def SVM(self, subcorpora, **kwargs):
        return self.classify(subcorpora, SVC, **kwargs)
    # Returns a naive Bayesian classifier. `name` should one of 'gaussian', 
    # 'multinomial', or 'bernoulli'.
    def naive_bayes(self, subcorpora, name = 'gaussian', **kwargs):
        if name == 'gaussian':
            classifier = GaussianNB(**kwargs)
        elif name == 'multinomial':
            classifier = MultinomialNB(**kwargs)
        elif name == 'bernoulli':
            classifier = BernoulliNB(**kwargs)
        else:
            raise RuntimeError, 'Unknown Bayesian strategy ' + name
        return self.classify(subcorpora, classifier, **kwargs)
    # Returns a decision tree classifier.
    def decision_tree(self, subcorpora, **kwargs):
        return self.classify(subcorpora, DecisionTreeClassifier, **kwargs)
    # ======= CLUSTERING =======
    # Perform clustering; return both the clustering object as well as the
    # predicted labels for all of the documents in the given subcorpus.
    def cluster(self, subcorpus, cluster_fn, **kwargs):
        cluster = cluster_fn(**kwargs)
        X = self.get_subcorpus(subcorpus)
        labels = cluster.fit_predict(X)
        return (cluster, labels)
    def kmeans(self, subcorpus, n_jobs = -1, **kwargs):
        return self.cluster(subcorpus, KMeans, n_jobs = n_jobs, **kwargs)
    def spectral(self, subcorpus, **kwargs):
        return self.cluster(subcorpus, SpectralClustering, **kwargs)
    def hierarchical(self, subcorpus, **kwargs):
        return self.cluster(subcorpus, Ward, **kwargs)
    def dbscan(self, subcorpus, **kwargs):
        return self.cluster(subcorpus, DBSCAN, **kwargs)
    # ======= LSA =======
    # Performs LSA on the set of feature vectors, mapping to a semantic space 
    # of lower dimensionality. This is useful to increase the efficiency of 
    # some algorithms, though it comes with an important drawback: in mapping
    # to a space of lower dimensionality, we lose information about the 
    # original feature space itself. This is reflected in that the
    # feature_names_available attribute of the corpus object will be set to
    # False after this method is run. If you would like to keep the original
    # vectors with their feature name mappings, make a copy of the corpus 
    # object.
    # Keyword arguments are passed directly to the ScikitLearn TruncatedSVD
    # constructor.
    def LSA(self, **kwargs):
        svd = TruncatedSVD(**kwargs)
        self.vecs = svd.fit_transform(self.vecs)
        self.feature_names_available = False
        self.feature_names_unavailable_reason = "LSA"
        return LSA

    # TODO Add gensim support
        

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
