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

# TODO: see if the "masked" order can be made to work with autocorrelation.
#       otherwise, I think I should drop it (at least for a while). It may
#       but useful nonetheless for some applications, such as 3-tap filters
#       but then may be made AGAIN available under a different name (mask ?)
#       and only for the covariance method if it's the only one that works.

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

# TODO: a with some values set to 0. Return the full vector instead (with 0 in
#       the appropriate places). Would make the return value structure more 
#       orthogonal to the computation options, and a2k a no-brainer). Then,
#       how do we specify the coefficients that are 0 ? Use a convention similar
#       to numpy mask arrays ? for example set mask=[1, 0, 0, 0] if you want
#       a 4-order prediction with a_1 = 0 ? Then order becomes optional ...
#       Other possible conventions are "0" instead of "1" to denote the locations
#       of the zeros, or a list of the zero coefficients (or the opposite). Try
#       one convention, then try it. Maybe multiple conventions ? [True, False,
#       False, False] instead or [1, 0, 0, 0] (or the opposite) ? With the
#       opposite, u say "Yes" for all coefficients that you have the right
#       to use for the prediction. The term "mask" is ambiguous in this respect ...
#       Rk: the convention with True for "possibly non zero" allows to select
#       easily the non-zero coefficients from the lacunary vector a.

def a2k(a):
    # source: Rabiner, Schafer
    # Q: also works for lacunary a's ? We don't care, just give it the full
    #    a sequence, with the zeros, and it will work. (And no, it doesn't work)
    m = len(a)
    A = np.zeros((m,m))
    A[m-1,:] = a
    for i in np.arange(m-1, 0, -1): # m-1, m-2, ..., 1
        ki = A[i, i]
        js = np.arange(i)
        # Rk: the external setting of numpy.seterr matters when |ki| = 1.0.
        A[i-1,js] = (A[i,js] + ki * A[i,i-1-js]  )/ (1 - ki * ki)
    return np.diagonal(A)

def k2a(k):
    # source: Rabiner, Schafer
    m = len(k)
    A = np.diag(k)
    for i in np.arange(m-1):
        js = np.arange(i+1)
        A[i+1,js] = A[i,js] - k[i+1] * A[i,i-js]
    return A[m-1,:].copy()

def lp(x, order, method="covariance", algo=None, window=None, returns="a"):
    """
    Linear Predictor Coefficients

    Arguments
    ---------

      - `x`: the time series, a sequence of floats.
 
      - `order`: prediction order if `order` is an `int`, 
        otherwise the list of non-zero indices of the predictor coefficients:
        the selection of `order = m` is therefore a shortcut for 
        `order = [1, 2, ..., m]`.

      - `method`: `"covariance"` or `"autocorrelation"` (default: "covariance"). 
        The short names `"cv"` and `"ac"` are also valid.
      
      - `window`: function, optional: a window applied to the signal.

      - `returns`: a sequence of strings or comma-separated string of variable names.
        When `returns` is a single string identifier, without a trailing comma, the
        value with this name is returned ; otherwise the named value(s) is (are) 
        returned as a tuple. Defaults to "a".

    Returns
    -------
    
    The set of returned values is selected by the `returns` argument among:

      - `a`: the array of prediction coefficients `[a_1, ..., a_m]`.

        The predicted value `y[n]` of `x[n]` should be computed with:
 
            y[n] = a_1 * x_[n-1] + ... + a_m * x[n-m]

      - `k`: the array of reflection coefficients `[k_1, ..., k_m]`.

    """

    # Rk: method is covariance or autocorrelation, algo = "LS" (least squares)
    #     or "LTZ" (Levinson-Trench-Zohar). "LS" is applicable to both methods
    #     but "LTZ" only to autocorrelation.

    unwrap_returns = False
    if isinstance(returns, str):
        returns_args = [name.strip() for name in returns.split(',')]
        if len(returns_args) == 1:
            unwrap_returns = True
        if len(returns_args) >= 1 and not returns_args[-1]: # trailing comma
            returns_args = returns_args[:-1]
    else:
        returns_args = returns
    for name in returns_args:
        if name not in ("a", "k"):
            raise ValueError("invalid return value {0!r}".format(name))


    x = np.array(x, copy=False) # TODO: check 1d.

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

    if method in ("cv", "covariance"):
        pass
    elif method in ("ac", "autocorrelation"):
        x = np.r_[np.zeros(m), x, np.zeros(m)]
    else:
        raise ValueError("invalid method name: {0!r}".format(method))

    # temporary (as long as LTZ is not implemented)
    assert algo is None or algo in ("ls", "least squares") 

    if m >= len(x):
        raise ValueError("the prediction order is larger than the length of x")

    x = np.ravel(x)
    n = len(x)

    logfile.debug("x: {x}")
    logfile.debug("n: {n}")

    A = np.array([x[m - order + i] for i in range(n-m)])
    b = np.ravel(x[m:n])

    logfile.debug("A: {A}")
    logfile.debug("b: {b}")

    a_, _, _ ,_ = np.linalg.lstsq(A, b) # can't trust the residues (may be [])

    # create the lacunary a from the dense one
    a = np.zeros(m)
    a[order-1] = a_

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

    if "k" in returns_args:
        k = a2k(a)

    returns = tuple([locals()[arg] for arg in returns_args])
    if unwrap_returns:
        returns = returns[0]
    return returns

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
    ...     a = lp(x, n, method="autocorrelation")
    ...     stable.append(all(abs(np.roots(a)) < 1))
    >>> all(stable)
    True

Compute predictor with selected non-zero coefficients: 

    >>> x = [-1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]
    >>> lp(x, 2) # doctest: +NUMBER
    [0.0, -1.0]
    >>> lp(x, [2]) # doctest: +NUMBER
    [0.0, -1.0]
    >>> lp(x, 4)[1::2] # doctest: +NUMBER
    [-0.5, 0.5]
    >>> lp(x, [4]) # doctest: +NUMBER
    [0.0, 0.0, 0.0, 1.0]
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

