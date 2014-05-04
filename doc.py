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

# A document is an in-memory representation of a text stream. The constructor
# accepts a "stream" (some iterable sequence of features) and uses the
# collections.Counter object in the standard library to compute a mapping from
# features to frequencies. Unlike a normal collections.Counter object, however,
# documents are not mutable after they have been constructed. A document
# also has an optional "info" field which can be used to "label" the document
# in some way.
class Document:
    def __init__(self, stream = [], info = None):
        self.__counter = collections.Counter(stream)
        self.num_features = sum(self.__counter.values())
        self.info = info
    def __add__(self, other):
        return Document(self.__counter + other.__counter)
    def __getitem__(self, key):
        return self.__counter[key]
    def __iter__(self):
        return iter(self.__counter)
    def __len__(self):
        return len(self.__counter)
    def elements(self):
        return self.__counter.elements()
    def items(self):
        return self.__counter.items()
    def most_common(self, n = None):
        return self.__counter.most_common(n)
    @staticmethod
    def from_filename(fname, streamfn = words, info = None, **kwargs):
        return Document.from_file(open(fname, 'r'), streamfn = streamfn, 
                                  info = info, **kwargs)
    @staticmethod
    def from_file(f, streamfn = words, info = None, **kwargs):
        return Document(streamfn(f, **kwargs), info = info)
    @staticmethod
    def from_string(s, streamfn = words, info = None, **kwargs):
        return Document.from_file(FilelikeString(s), streamfn = streamfn,
                                  info = info, **kwargs)
