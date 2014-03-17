#!/usr/bin/env python
# coding: utf-8
"""
Linear Prediction Toolkit
"""

#
# Dependencies
# ------------------------------------------------------------------------------
# 
#   - the standard Python 2.7 library,
#   - the [NumPy][] and [SciPy][] libraries,
#   - the [logfile][] module,
# 
# [NumPy]: http://numpy.scipy.org/
# [SciPy]: http://scipy.org/
# [logfile]: https://github.com/boisgera/logfile
#

# Python 2.7 Standard Library
import argparse
import doctest
import inspect
import math
import os
import sys

# Third-Party Librairies
import numpy as np
from scipy.linalg import *

# Digital Audio Coding
import numtest
import logfile

#
# Metadata
# ------------------------------------------------------------------------------
#

__author__ = u"Sébastien Boisgérault <Sebastien.Boisgerault@mines-paristech.fr>"
__version__ = u"trunk"
__license__ = "MIT License"

#
# Linear Prediction -- Wiener-Hopf Filter
# ------------------------------------------------------------------------------
#

# TODO: have a look at <http://thomas-cokelaer.info/software/spectrum/html/contents.html>
#       (the focus is on spectrum estimation, but it implements lp algs nonetheless).

# TODO: study integration of Levison-Durbin, maybe Burg, etc.

# TODO: get rid of zero_padding in favor of a method name: "covariance" or
#       "autocorrelation" to begin with.

# TODO: when zero-padding is False, should check that order is not too big wrt
#       the window or raise an error.

# TODO: to make a 3-tap ltp prediction, support for observation window in lp
#       would be handy.

# Rk: window is probaly not worth it, it's an orthogonal concern.

# order: the overloading for forbidden indexes kind of sucks.
#        A sequence of orders means instead that you want to
#        run SEVERAL linear prediction of several orders, which
#        makes sense with recursive algorithms (think Levinson-Durbin).

# TODO: (short-term): support only ONE numeric order, later a sequence of
#       numeric orders (to be recursive algo friendly and output several
#       a vectors or k vectors at once).

# TODO: transfer levinson algorithm here.

# TODO: support "output" string based on "a" and "k". Implement a to k and
#       k to a converters.

def lp(x, order, zero_padding=False, window=None):
    """
    Wiener-Hopf Predictor.

    Arguments
    ---------

      - `x`: the signal x, a sequence of floats.
 
      - `order`: prediction order if `order` is an `int`, 
        otherwise the list of non-zero indices of the predictor coefficients:
        the selection of `order = m` is therefore a shortcut for 
        `order = [1, 2, ..., m]`.

      - `zero_padding`: `True` for the covariance method 
         or `False` -- the default -- for the autocorrelation method.
      
      - `window`: function, optional: a window applied to the signal.


    Returns
    -------
    
      - `a`: the array of prediction coefficients `[a_1, ..., a_m]`.

        The predicted value `y[n]` of `x[n]` should be computed with:
 
            y[n] = a_1 * x_[n-1] + ... + a_m * x[n-m]
    """

    x = np.array(x, copy=False)

    if isinstance(order, int):
        m = order
        order = np.arange(1, m + 1)
    else:
        m = order[-1]
        order = np.array(order, copy=False)

    if order.size == 0:
        return np.array([])

    if window:
        signal = window(len(x)) * x

    if zero_padding: # select autocorrelation method instead of covariance
        x = np.r_[np.zeros(m), x, np.zeros(m)]

    if m >= len(x):
        raise ValueError("the prediction order is larger than the length of x")

    x = np.ravel(x)
    n = len(x)

    logfile.debug("x: {x}")
    logfile.debug("n: {n}")

    # Issue when order >= len(signal), investigate. Force zero-padding ?

    A = np.array([x[m - order + i] for i in range(n-m)])
    b = np.ravel(x[m:n])

    logfile.debug("A: {A}")
    logfile.debug("b: {b}")

    a, _, _ ,_ = np.linalg.lstsq(A, b) # can't trust the residues (may be [])

    logfile.debug("a: {a}")
    h = np.r_[1.0, -a]
    error = np.convolve(h, x)[m:-m] # error restricted to the error window
    # basically useless (windowing taken into account) unless you want to 
    # compute some sort of error.
    logfile.debug("error: {error}")

    try:
        config = numpy.seterr(all="ignore")
        relative_error = np.sqrt(np.sum(error**2) / np.sum(x[m:-m]**2))
        logfile.debug("relative error: {relative_error}") 
    finally:
        numpy.seterr(**config)

    return a

#
# Unit Tests
# ------------------------------------------------------------------------------
#

def test_predictor():
    """
Test the predictor function on known results 

    >>> x = [1.0, 1.0]
    >>> lp(x, 0) # doctest: +NUMBER
    []

    >>> x = [1.0, 1.0]
    >>> lp(x, 1) # doctest: +NUMBER
    [1.0]

    >>> x = [2.0, 2.0, 2.0, 2.0]
    >>> lp(x, 1) # doctest: +NUMBER
    [1.0]

    >>> x = [1.0, -1.0, 1.0, -1.0]
    >>> lp(x, 1) # doctest: +NUMBER
    [-1.0]

    >>> x = [1.0, 2.0, 3.0, 4.0]
    >>> lp(x, 2) # doctest: +NUMBER
    [2.0, -1.0]

    >>> x = [0.0, 1.0, 0.0, -1.0]
    >>> lp(x, 1) # doctest: +NUMBER
    [0.0]

    >>> x = [0.0, 1.0, 0.0, -1.0]
    >>> lp(x, 2) # doctest: +NUMBER
    [0.0, -1.0]

    >>> x = [1.0, 2.0, 4.0, 8.0, 16.0]
    >>> lp(x, 1) # doctest: +NUMBER
    [2.0]

Test the stability of the prediction filter autocorrelation method

    >>> x = [1.0, 2.0, 4.0, 8.0, 16.0]
    >>> stable = []
    >>> for n in [1, 2, 3, 4, 5]:
    ...     a = lp(x, n, zero_padding=True)
    ...     stable.append(all(abs(np.roots(a)) < 1))
    >>> all(stable)
    True

Compute predictor with selected non-zero coefficients: 

    >>> x = [-1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
    >>> lp(x, 2) # doctest: +NUMBER
    [0.0, -1.0]
    >>> lp(x, [2]) # doctest: +NUMBER
    [-1.0]
    >>> lp(x, 4)[1::2] # doctest: +NUMBER
    [-0.5, 0.5]
    >>> lp(x, [4]) # doctest: +NUMBER
    [1.0]
    """

def test(verbose=True):
    """
    Run the unit tests
    """
    import doctest
    return doctest.testmod(verbose=verbose)

#
# Command-Line Interface
# -----------------------------------------------------------------------------
#

def main(args):
    "Command-line interface entry point"
    description = "Run the linear prediction toolkit test suite."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", 
                        action  = "count", 
                        default = 0,
                        help    = "display more information")
    parser.add_argument("-s", "--silent",
                        action  = "count", 
                        default = 0,
                        help    = "display less information")
    args = parser.parse_args()
    verbose = (args.verbose - args.silent) > 0
    test_results = test(verbose=verbose)
    sys.exit(test_results.failed)

if __name__ == "__main__":
    main(sys.argv[1:])
