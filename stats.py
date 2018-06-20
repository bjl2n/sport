import constants as const
import math
import numpy as np
import scipy.stats as sps


# noinspection PyTypeChecker
def mean(distribution_info):
    """Returns the mean of the given distribution.
    @type distribution_info: list
    @param distribution_info: A list of information related to the
    distribution.  The first entry is the name (str) of the distribution,
    and any other elements of the list correspond to parameters of the
    distribution.
    @rtype : float
    """
    dist_name = distribution_info[0].lower()
    assert dist_name in const.DISTRIBUTIONS, \
        'Mean of %s distribution not defined' % (distribution_info[0])

    if dist_name == 'constant':
        return distribution_info[1]

    if dist_name == 'beta':
        alpha = float(distribution_info[1])
        beta = distribution_info[2]
        return alpha / (alpha + beta)

    if dist_name == 'gamma':
        shape = distribution_info[1]
        scale = distribution_info[2]
        return shape * scale

    if dist_name == 'triangular':
        left = distribution_info[1]
        mode = distribution_info[2]
        right = distribution_info[3]
        return (left + mode + right) / 3.0

    if dist_name == 'uniform':
        low = distribution_info[1]
        high = distribution_info[2]
        return (low + high) / 2.0


# TODO finish defining the minimums
# noinspection PyTypeChecker
def minimum(distribution_info):
    """Returns the minimum of the given distribution.
    @type distribution_info: list
    @param distribution_info: A list of information related to the
    distribution.  The first entry is the name (str) of the distribution,
    and any other elements of the list correspond to parameters of the
    distribution.
    @rtype : float
    """
    dist_name = distribution_info[0].lower()
    assert dist_name in const.DISTRIBUTIONS, \
        'Minimum of %s distribution not defined' % (distribution_info[0])

    if dist_name == 'constant':
        return distribution_info[1]

    if dist_name == 'beta':
        assert False, 'Minimum of %s distribution not defined' % (distribution_info[0])

    if dist_name == 'gamma':
        assert False, 'Minimum of %s distribution not defined' % (distribution_info[0])

    if dist_name == 'triangular':
        left = distribution_info[1]
        return left

    if dist_name == 'uniform':
        low = distribution_info[1]
        return low


# noinspection PyTypeChecker
def variance(distribution_info):
    """Returns the variance of the given distribution.
    @type distribution_info: list
    @param distribution_info: A list of information related to the
    distribution.  The first entry is the name (str) of the distribution,
    and any other elements of the list correspond to parameters of the
    distribution.
    """
    dist_name = distribution_info[0].lower()
    assert dist_name in const.DISTRIBUTIONS, \
        'Variance of %s distribution not defined' % (distribution_info[0])

    if dist_name == 'constant':
        return 0

    if dist_name == 'beta':
        alpha = float(distribution_info[1])
        beta = distribution_info[2]
        return (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))

    if dist_name == 'gamma':
        shape = distribution_info[1]
        scale = distribution_info[2]
        return shape * (scale ** 2)

    if dist_name == 'triangular':
        left = distribution_info[1]
        mode = distribution_info[2]
        right = distribution_info[3]
        return ((left ** 2) + (right ** 2) + (mode ** 2) -
                (left * right) - (left * mode) - (right * mode)) / 18.0

    if dist_name == 'uniform':
        low = distribution_info[1]
        high = distribution_info[2]
        return ((high - low) ** 2) / 12.0


def standard_dev(distribution_info):
    """Returns the standard deviation of the given distribution.
    @type distribution_info: list
    @param distribution_info: A list of information related to the
    distribution.  The first entry is the name (str) of the distribution,
    and any other elements of the list correspond to parameters of the
    distribution.
    """
    dist_name = distribution_info[0].lower()
    assert dist_name in const.DISTRIBUTIONS, \
        'Standard deviation of %s distribution not defined' % (distribution_info[0])

    if dist_name == 'constant':
        return 0

    if dist_name == 'beta':
        return math.sqrt(variance(distribution_info))

    if dist_name == 'gamma':
        return math.sqrt(variance(distribution_info))

    if dist_name == 'triangular':
        return math.sqrt(variance(distribution_info))

    if dist_name == 'uniform':
        return math.sqrt(variance(distribution_info))


# noinspection PyPep8Naming
def mean_and_CI_half_width(data, alpha=0.05):
    """Give a list of ints or floats, calculate the mean and (1-alpha)% CI.
    @type data: list[float]
    @type alpha: float
    @return: mean, CI_half_width
    """
    if data:
        n = len(data)
        standard_error = sps.sem(data)
        confidence_level = 1 - (alpha / 2.0)
        CI_half_width = standard_error * sps.t.ppf(confidence_level, n - 1)

        return np.mean(data), CI_half_width
    else:
        assert False, 'No data passed in'


def percentile(distribution_info, pctile):
    """Returns the percentile of the given distribution.
    @type distribution_info:
    @type pctile: float
    @rtype: float
    """
    dist_name = distribution_info[0].lower()
    assert dist_name in const.DISTRIBUTIONS, \
        '%s distribution not defined' % (distribution_info[0])

    if dist_name == 'beta':
        assert False, 'Percentile of %s distribution not defined' % (distribution_info[0])

    if dist_name == 'gamma':
        assert False, 'Percentile of %s distribution not defined' % (distribution_info[0])

    if dist_name == 'triangular':
        assert False, 'Percentile of %s distribution not defined' % (distribution_info[0])

    if dist_name == 'uniform':
        params = [distribution_info[1], distribution_info[2] - distribution_info[1]]
        return getattr(sps, dist_name)(*params).ppf(pctile)
