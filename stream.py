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
from philoyore.io import words, FileLikeString
from philoyore.features import FeatureSet

# A StreamSet represents a collection of "streams", a "stream" being data from
# some input source. The input source is "tokenized" in some way so that the
# input stream is converted into a collection of feature-count pairs, where a
# "feature" is usually a word, lemma, n-gram, n-lemma, and so on and so forth.
# The stream set can be manipulated at a high level by adding or removing
# features from it; we use the collections.Counter object in the Python standard
# library to make this rather easy.
class StreamSet:
    # The initializer accepts a list of `streams` which are used to populate the
    # streamset. Other constructor functions in this module can be used
    # to populate StreamSets in different ways. 
    def __init__(self, streams, names = None):
        if len(streams) <= 0:
            raise RuntimeError, "The input stream list must not be empty"
        self.streams = streams
        self.total = sum(streams, collections.Counter)
        self.names = names if names is not None \
                     else [None for _ in range(len(self))]
    def __len__(self):
        return len(self.streams)
    def __getitem__(self, key):
        return self.streams[key]
    def __delitem__(self, key):
        deleted = self.streams[key]
        del self.streams[key]
        if self.names is not None:
            del self.names[key]
        if isinstance(delete, list):
            for d in deleted:
                self.total -= d
        else:
            self.total -= deleted
    def __iter__(self):
        return iter(self.streams)
    def append(self, stream, name = None):
        self.streams.append(stream)
        if self.names is not None:
            self.names.append(name)
        self.total += stream
    def __add__(self, other):
        return StreamSet(self.streams + other.streams, 
                         names = self.names + other.names)
    def features(self, **kwargs):
        return FeatureSet(self, **kwargs)
    def clone(self):
        return StreamSet(self.streams[:], names = self.names[:])

def stream_from_filename(fname, streamfn = words):
    f = open(fname, 'r')
    return stream_from_file(f, streamfn)

def stream_from_file(f, streamfn = words):
    return collections.Counter(streamfn(f))

def stream_from_string(s, streamfn = words):
    return stream_from_file(FileLikeString(s), streamfn)

def stream_from_dict(d):
    return collections.Counter(d)

def set_from_filenames(fnames, streamfn = words):
    return StreamSet([stream_from_filename(f, streamfn) for f in fnames],
                     names = fnames)

def set_from_files(fs, streamfn = words):
    def getname(f):
        try:
            name = f.name
        except AttributeError:
            name = None
        return name
    return StreamSet([stream_from_file(f, streamfn) for f in fs],
                     names = [getname(f) for f in fs])

def set_from_strings(ss, streamfn = words):
    cutlen = 15
    def processword(s):
        if len(s) <= cutlen:
            return s
        else:
            return s[0:15] + '...'
    return StreamSet([stream_from_string(s, streamfn) for s in ss],
                     names = [processword(s) for s in ss])

def set_from_dicts(ds):
    return StreamSet([stream_from_dict(d) for d in ds])
