
# Third-Party
import numpy as np

# Digital Audio
import lp

# Reference: http://www.scribd.com/doc/29323516/38/The-Levinson%E2%80%93Durbin-Recursion
# (Also for the corresponding lattice structures). Actually, the reference seems
# to be nice for speech processing in general.

# ------------------------------------------------------------------------------

# TODO: Study a <---> k converters.


# TODO: Before an integration of levinson and lp both cods:
#       Rethink the optional arguments in lp such as 
#       zero-padding. Grow up and accept the standard "autocorrelation" and
#       "covariance" method ? Or fuse the method name and the algorithm ?
#       (i.e. accept "Levinson-Durbin" ? "Burg" ? That kind of things ?) 
#       Can we adapt levinson to handle "places" (order 
#       not a number, but a sequence of indices that can be non-zero).
#       To see that, I have to understand the proof of the Levinson
#       Durbin algorithms.

def levinson(x, order):
    x = np.array(x, copy=False)
    N = len(x)
    r = np.convolve(x, x[::-1])[N-1:] # r[i], for i up to N-1.
                                      # if more is needed, pad with zeros.

    E = np.zeros(order + 1)
    k = np.zeros(order)
    a = np.zeros((order, order))

    E[0] = r[0]
    for i in np.arange(1, order + 1):
        k[i-1] = (r[i] - np.dot(a[i-2,0:i-1], r[i-1:0:-1])) / E[i-1]
        a[i-1,i-1] = k[i-1]
        for j in np.arange(i-1):
            a[i-1,j] = a[i-2,j] - k[i-1] * a[i-2, i-2-j]
        E[i] = (1.0 - k[i-1]*k[i-1]) * E[i-1]

    a = a[-1, :]

    # Should we return the full a ? 
    # Rk: return (fir, ar) dl, il filters in the dictionary ?
    #     return ONE of them for analysis, ONE for synthesis, under
    #     the "analysis" and "synthesis" keys ?
    return dict(a=a, k=k)


