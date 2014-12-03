#!/usr/bin/env python
# coding: utf-8

# Python 2.7 Standard Library
import doctest

# Extra Libraries
import numtest

# This package
import audio.lp

#
# Test Runner
# ------------------------------------------------------------------------------
#

# support for `python setup.py test`:
suite = doctest.DocTestSuite(audio.lp)

if __name__ == "__main__":
    doctest.testmod(audio.lp)

