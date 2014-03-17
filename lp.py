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
#   - the `logger` and `script` modules from the digital audio coding project,
# 
# [NumPy]: http://numpy.scipy.org/
# [SciPy]: http://scipy.org/
# [lsprofcalltree]: http://people.gnome.org/~johan/lsprofcalltree.py
#

# Python 2.7 Standard Library
import doctest
import inspect
import math
import os
import sys

# Third-Party Librairies
from numpy import *
from scipy.linalg import *

# Digital Audio Coding
import numtest
import logger
import script

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

# TODO: study integration of Levison-Durbin, maybe Burg, etc.

# TODO: when zero-padding is False, should check that order is not too big wrt
#       the window or raise an error.

# TODO: to make a 3-tap ltp prediction, support for observation window in lp
#       would be handy.

# Rk: window is probaly not worth it, it's an orthogonal concern.

# order: the overloading for forbidden indexes kind of sucks.
#        A sequence of orders means instead that you want to
#        run SEVERAL linear prediction of several orders, which
#        makes sense with recursive algorithms (think Levinson-Durbin).

@logger.tag("lp.lp")
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

    x = array(x, copy=False)

    if isinstance(order, int):
        m = order
        order = arange(1, m + 1)
    else:
        m = order[-1]
        order = array(order, copy=False)

    if order.size == 0:
        return array([])

    if window:
        signal = window(len(x)) * x

    if zero_padding: # select autocorrelation method instead of covariance
        x = concatenate((zeros(m), x, zeros(m)))

    if m >= len(x):
        raise ValueError("the prediction order is larger than the length of x")

    x = ravel(x)
    n = len(x)

    print >> logger.debug, "x:", x
    print >> logger.debug, "n:", n

    # Issue when order >= len(signal), investigate. Force zero-padding ?

    A = array([x[m - order + i] for i in range(n-m)])
    b = ravel(x[m:n])

    print >> logger.debug, "A:", A
    print >> logger.debug, "b:", b

    a, _, _ ,_ = linalg.lstsq(A, b) # can't trust the residues (may be [])

    print >> logger.debug, "a:", a
    h = concatenate(([1.0], -a))
    print >> logger.debug, "h:", h
    error = convolve(h, x)[m:-m] # error restricted to the error window
    # basically useless (windowing taken into account) unless you want to 
    # compute some sort of error.
    print >> logger.debug, "error:", error

    try:
        config = numpy.seterr(all="ignore")
        relative_error = sqrt(sum(error**2) / sum(x[m:-m]**2))
        print >> logger.debug, "relative error:", relative_error 
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
    ...     stable.append(all(abs(roots(a)) < 1))
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

def help():
    """
Return the following message:

    Run the linear prediction test suite.    

    usage:
        python lp.y [OPTIONS]

    options: -h, --help ........................ display help message and exit,
             -t, --test ........................ run the module self tests,
             -v, --verbose ..................... verbose mode.
"""
    return "\n".join(line[4:] for line in inspect.getdoc(help).splitlines()[2:])

def main(args):
    """Command-line interface entry point"""
    options, args = script.parse("help test verbose", args)

    if options.help:
        print help()
        sys.exit(0)

    verbose = bool(options.verbose)
    if options.test:
        verbose = bool(options.verbose)
        test_results = test(verbose=verbose)
        sys.exit(test_results.failed)
    else:
        print help()
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])

