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
from philoyore.io import words, FilelikeString
from philoyore.features import FeatureSet
from philoyore.doc import Document

# A Corpus represents a collection of "documents", a "document" being data from
# some input source. The input source is "tokenized" in some way so that the
# input stream is converted into a collection of feature-count pairs, where a
# "feature" is usually a word, lemma, n-gram, n-lemma, and so on and so forth.
# The corpus can be manipulated at a high level by adding or removing
# documents from it; we use the collections.Counter object in the Python 
# standard library to make this rather easy.
class Corpus:
    # The initializer accepts a list of `documents` which are used to populate 
    # the object. Other constructor functions in this module can be used
    # to populate Corpora in different ways.
    def __init__(self, docs = []):
        self.docs = docs
        self.total = sum(docs, Document())
    def __add__(self, other):
        return Corpus(self.docs + other.docs)
    def __delitem__(self, key):
        deleted = self.docs[key]
        del self.docs[key]
        if isinstance(delete, list):
            for d in deleted:
                self.total -= d
        else:
            self.total -= deleted
    def __getitem__(self, key):
        return self.docs[key]
    def __iter__(self):
        return iter(self.docs)
    def __len__(self):
        return len(self.docs)
    def append(self, doc):
        self.docs.append(doc)
        self.total += doc
    def clone(self):
        return Corpus(self.docs[:])
    def features(self, **kwargs):
        return FeatureSet(self, **kwargs)
    def filter(self, fn):
        self.docs = filter(fn, self.docs)
        self.total = sum(self.docs, Document())
    @staticmethod
    def from_filenames(fnames, streamfn = words, info = None, **kwargs):
        info = info if info is not None else fnames
        return Corpus([Document.from_filename(fname, streamfn = streamfn, 
                                              info = i, **kwargs) \
                           for fname, i in zip(fnames, info)])
    @staticmethod
    def from_files(fs, streamfn = words, info = None, **kwargs):
        info = info if info is not None else [f.name for f in fs]
        return Corpus([Document.from_file(f, streamfn = streamfn, 
                                          info = i, **kwargs) \
                           for f, i in zip(fs, info)])
    @staticmethod
    def from_strings(ss, streamfn = words, info = None, **kwargs):
        info = info if info is not None else [s[0:15] for s in ss]
        return Corpus([Document.from_string(s, streamfn = streamfn, 
                                            info = i, **kwargs) \
                           for s, i in zip(ss, info)])
