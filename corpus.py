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

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
    def __init__(self, groups, strategy = 'count', input = 'filename', 
                 **options):
        all_groups = sum(groups, [])
        if strategy == 'count':
            self.vectorizer = CountVectorizer(input = input, **options)
        elif strategy == 'tf-idf':
            self.vectorizer = TfidfVectorizer(input = input, **options)
        else:
            raise RuntimeError, 'Unrecognized vectorization strategy ' + \
                strategy
        self.vecs = self.vectorizer.fit_transform(all_groups)

# Convert a list of files to a corpus with one subcorpus.
def from_files(fs, **options):
    return Corpus([fs], input = 'file', **options)

# Convert a list of filenames to a corpus with one subcorpus.
def from_filenames(fns, **options):
    return Corpus([fns], **options)

# Convert a list of strings to a corpus with one subcorpus.
def from_strings(ss, **options):
    return Corpus([ss], input = 'content', **options)

# Convert a list of lists of files to a corpus.
def from_file_lists(fss, **options):
    return Corpus(fss, input = 'file', **options)

# Convert a list of lists of fileneames to a corpus (the default behavior of 
# the constructor).
def from_filename_lists(fnss, **options):
    return Corpus(fnss, **options)

# Convert a list of lists of strings to a corpus.
def from_string_lists(sss, *options):
    return Corpus(sss, input = 'content', **options)
