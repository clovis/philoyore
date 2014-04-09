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


# This file contains some utility methods which can be used to stream data from
# files in various ways. With philoyore, it is usually the case that we want
# to stream a file in "chunks", where a "chunk" can mean one of several
# different things (often a word or a sequence of words). All of the utility
# methods here implement streaming result sets; that is, they do NOT work
# by reading the entire file into memory and processing a large string at
# runtime, which could be prohibitive from a computational standpoint.

# philIO is, at its core, a tokenizing library; this file is therefore primarily
# concerned with converting files into sequences of tokens, where each token
# has the same shape. The "shape" of each token varies based on which method
# is called. The more interesting functionality of the library can be found
# in other files.

# Generate all the words of the given file (here, a "word" is defined as
# a sequence of characters surrounded on either side by spaces). Whitespace
# is stripped out. We do this by reading a smaller subsection of the file into
# memory and processing the in-memory buffer manually. There is a corner
# case to be aware of: it is rather likely that we will read in only part of
# the last word in the buffer. We correct this by saving the last-read word
# if we are uncertain it is complete and processing it on the next iteration.
# In the interest of future optimizations, setting the read size to be a 
# greater value than `buffer_size`  would result in fewer calls to the 
# system-level read function in exchange for more memory used at runtime. 
# Since I have a feeling IO costs will be non-trivial in the runtime of 
# Philoyore proper, it will be an interesting exercise to mess with that value 
# to see how performance changes.
def words(f):
    buffer_size = 1024
    next_buf = ""
    while True:
        current_buf = next_buf + f.read(buffer_size)
        if len(current_buf) == 0:
            break
        words = current_buf.split()
        # If the last character in the buffer is whitespace, we can be sure
        # the last word in the `words` array is complete. If it is not, we 
        # cannot, so we remove it from the `words` array and add it to the next
        # buffer so it can be processed at the next iteration.
        if current_buf[-1].isspace():
            next_buf = ""
        else:
            next_buf = words[-1]
            del words[-1]
        for word in words:
            yield word

# Generate a sequence of n-grams from the given file. For example, with an
# n-value of 2 and a file that contains the text "to be or not to be", the
# sequence of n-grams generated will be "to be", "be or", "or not", "not to",
# and "to be". N-grams are returned as tuples with length n. 
# The implementation leverages the words method in this same module.
def ngrams(f, n):
    if n <= 0:
        raise RuntimeError, "ngrams: n must be a positive integer"
    arr = []
    for word in words(f):
        arr.append(word)
        if len(arr) > n:
            del arr[0]
            yield tuple(arr)
        elif len(arr) == n:
            yield tuple(arr)
        else:
            continue

# Generate a sequence of bigrams from the given file. This is a convenience
# function that simply calls ngrams.
def bigrams(f):
    for bigram in ngrams(f, 2):
        yield bigram

# Generate a sequence of trigrams from the given file. This is a convenience
# function that simply calls ngrams.
def trigrams(f):
    for trigram in ngrams(f, 3):
        yield trigram

# The FileLikeString class is a convenience class that allows you to treat 
# strings as files. This is useful because the methods in the philIO package
# all take files as their arguments; that is, they do streaming IO on files.
# This is useful for our particular use case, where we are simply reading
# text from disk; however, the same architecture does not allow you to generate
# words or n-grams from strings that are already in memory, which may be 
# useful for debugging or testing. The FileLikeString class simulates a file
# by exposing a `read` method that acts like the `read` method of the built-in
# file class in Python. Then you can easily generate a series of n-grams from
# a string `s` like so:
# for ngrams(FilelikeString(s), 4):
#     ....
# The class does not expose any other methods besides `read` at this point in
# time.
class FilelikeString:
    def __init__(self, s):
        self.i = 0  # Where to start reading in the string next
        self.s = s 
    def read(self, n=-1):
        if n < 0:
            new_s = self.s[self.i:]
            self.i = len(self.s)
        else:
            new_s = self.s[self.i : self.i + n]
            self.i = min(self.i + n, len(self.s))
        return new_s
