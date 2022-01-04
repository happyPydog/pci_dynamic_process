r"""Functional interface"""
import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from scipy.optimize import newton
import scipy.integrate as integrate


############################
# Process Capability Index #
############################


def cp(sigma, USL, LSL):
    """Process capability index Cp."""
    return (USL - LSL) / (6 * sigma)


def cpu(mean, sigma, USL):
    """Process capability index Cpu."""
    return (USL - mean) / (3 * sigma)


def cpl(mean, sigma, LSL):
    """Process capability index Cpl."""
    return (mean - LSL) / (3 * sigma)


def cpk(mean, sigma, USL, LSL):
    """Process capability index Cpk."""
    return np.minimum(((USL - mean) / (3 * sigma)), ((mean - LSL) / (3 * sigma)))


def cpm(mean, sigma, USL, LSL, T):
    """Process capability index Cpm."""
    return (USL - LSL) / (6 * np.sqrt(sigma ** 2 + (mean - T) ** 2))


def cpmk(mean, sigma, USL, LSL, T):
    """Process capability index Cpmk."""
    return np.minimum((USL - mean), (mean - LSL)) / (
        3 * np.sqrt(sigma ** 2 + (mean - T) ** 2)
    )


def dynamic_cpk(mean, sigma, USL, LSL, k1, k2):
    """Process capability index dynamic Cpk"""
    return np.minimum(
        ((USL - (mean + k1 * sigma)) / (3 * sigma * k2)),
        (((mean - k1 * sigma) - LSL) / (3 * sigma * k2)),
    )


def ncppm(cpk):
    """Calculate non-conformities part per million."""
    return 2 - 2 * norm.cdf(3 * cpk)


def million_nc(cpk):
    """Calculate Million non-conformities."""
    return (2 - 2 * norm.cdf(3 * cpk)) * 10 ** 6


###############################################################################
# Factors for Constructing Variables Control Charts                           #
# (Montegomery Introduction-to-statistical-quality-control-7th P.720)         #
###############################################################################

# Factor for center line
def c4(n):
    """Factor Variables c4."""
    assert np.all(n >= 2), "Sample size n must >= 2."
    return np.sqrt(2 / (n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))


def d2(n):
    """Factor Variables d2."""
    return integrate.quad(_w_first_moment, 0, np.inf, args=(n))[0]


# Factors for Control Limits
def A(n):
    """Factor Variables A."""
    return 3 / np.sqrt(n)


def A2(n, d2):
    """Factor Variables A2."""
    return 3 / (d2 * np.sqrt(n))


def A3(n, c4):
    """Factor Variables A3."""
    return 3 / (c4 * np.sqrt(n))


def B3(n):
    """Factor Variables c4."""
    return np.maximum(0, 1 - 3 / c4(n) * np.sqrt(1 - c4(n) ** 2))


def B4(n):
    """Factor Variables B4."""
    return 1 + 3 / c4(n) * np.sqrt(1 - c4(n) ** 2)


def B5(n):
    """Factor Variables B5."""
    return np.maximum(0, c4(n) - 3 * np.sqrt(1 - c4(n) ** 2))


def B6(n):
    """Factor Variables B6."""
    return c4(n) + 3 * np.sqrt(1 - c4(n) ** 2)


def d3(w_square, d2):
    """Factor Variables d3."""
    return np.sqrt(w_square - d2 ** 2)


def D1(d2, d3):
    """Factor Variables D1."""
    return np.maximum(0, d2 - 3 * d3)


def D2(d2, d3):
    """Factor Variables D2."""
    return d2 + 3 * d3


def D3(d2, d3):
    """Factor Variables D3."""
    return np.maximum(0, 1 - 3 * d3 / d2)


def D4(d2, d3):
    """Factor Variables D4."""
    return 1 + 3 * d3 / d2


def w_square(n):
    return integrate.quad(_w_second_moment, 0, np.inf, args=(n))[0]


def _w_first_moment(w, n):
    return 1 - w_cdf(w, n)


def _w_second_moment(w, n):
    return 2 * w * (1 - w_cdf(w, n))


def w_cdf(w, n):
    return integrate.quad(_w_cdf, -np.inf, np.inf, args=(w, n))[0]


def _w_cdf(x, w, n):
    return n * (norm.cdf(x + w) - norm.cdf(x)) ** (n - 1) * norm.pdf(x)


##############################
# Relative Range CDF and PDF #
##############################


def R_right(n, x0=5, alpha=0.0027):
    """UCL for the (alpha/2) of R distribution."""
    return newton(_right, x0, args=[n, alpha])


def R_left(n, x0=2, alpha=0.0027):
    """UCL for the 1-(alpha/2) of R distribution."""
    return newton(_left, x0, args=[n, alpha])


def _right(UCL, n, alpha):
    return integrate.dblquad(
        range_pdf, -np.inf, np.inf, lambda r: UCL, lambda r: 10, args=[n]
    )[0] - (alpha / 2)


def _left(LCL, n, alpha):
    return integrate.dblquad(
        range_pdf, -np.inf, np.inf, lambda r: 0, lambda r: LCL, args=[n]
    )[0] - (alpha / 2)


def range_pdf(r, t, n):
    """Probability Distribution Function of Range.
    reference
    =========
    [Yield Assessment for Dynamic Etching Processes With Variance Change]:
    <https://ieeexplore.ieee.org/document/9187659>
    """
    return (
        (n * (n - 1))
        / ((2 * np.pi) ** (n / 2))
        * np.exp(-(t ** 2 + (r + t) ** 2) / 2)
        * integrate.quad(_norm, t, r + t)[0] ** (n - 2)
    )


def _norm(z):
    return np.exp(-(z ** 2) / 2)
