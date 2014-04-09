# This module is the heart of the philoyore library; its exact shape will
# come to light as time goes on, but for now we intend that this will be
# the primary user interface to whatever algorithms this library will expose.
# Methods that consume input datasets will usually take them as files; these
# files will probably be tokenized using the philIO (pio) library so that
# this module can process them. The caller may utilize the FilelikeString
# class in the philIO library to convert Python strings (or objects that act
# like Python strings) into file-like objects that will be properly parsed
# by the philIO utilities.

import philIO as pio
import numpy as np
import scipy, scipy.spatial

