import pandas as pd
import numpy as np
import re
import empyrical as ep
from . import data_processing as dp

def factor_weights(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False,
                   long_short='long_short'):
    """
    Computes asset weights by factor values and dividing by the sum of their
    absolute value (achieving gross leverage of 1). Positive factor values will
    results in positive weights and negative values in negative weights.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Should this computation happen on a long short portfolio? if True,
        weights are computed by demeaning factor values and dividing by the sum
        of their absolute value (achieving gross leverage of 1). The sum of
        positive weights will be the same as the negative weights (absolute
        value), suitable for a dollar neutral long-short portfolio
    group_adjust : bool
        Should this computation happen on a group neutral portfolio? If True,
        compute group neutral weights: each group will weight the same and
        if 'demeaned' is enabled the factor values demeaning will occur on the
        group level.
    equal_weight : bool, optional
        if True the assets will be equal-weighted instead of factor-weighted
        If demeaned is True then the factor universe will be split in two
        equal sized groups, top assets with positive weights and bottom assets
        with negative weights
    long_short : str
        when demeand is True, if long_short=='long_short',compute weight both 
        weight>0 and <0; if long_short=='long',compute weight only weight>0; 
        if long_short=='short',compute weight only weight<0

    Returns
    -------
    returns : pd.Series
        Assets weighted by factor value.
    """

    def to_weights(group, _demeaned, _equal_weight,_long_short):

        if _equal_weight:
            group = group.copy()

            if _demeaned:
                # top assets positive weights, bottom ones negative
                group = group - group.median()

            negative_mask = group < 0
            group[negative_mask] = -1.0
            positive_mask = group > 0
            group[positive_mask] = 1.0

            if _demeaned:
                # positive weights must equal negative weights
                if negative_mask.any():
                    group[negative_mask] /= negative_mask.sum()
                if positive_mask.any():
                    group[positive_mask] /= positive_mask.sum()

        elif _demeaned:
            if _long_short=='long_short':
                group = group - group.mean()
            elif _long_short=='long':
                group = group - group.mean()
                group=(group*0+group[group>0]).replace(np.nan,0)
            elif _long_short=='short':
                group = group - group.mean()
                group=(group*0+group[group<0]).replace(np.nan,0)

        return group / group.abs().sum()

    grouper = [factor_data.index.get_level_values('date')]
    if group_adjust:
        grouper.append('group')

    weights = factor_data.groupby(grouper)['factor'] \
        .apply(to_weights, demeaned, equal_weight,long_short)

    if group_adjust:
        weights = weights.groupby(level='date').apply(to_weights, False, False,long_short)

    return weights


def factor_returns(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False,
                   by_asset=False,
                   long_short='long_short'):
    """
    Computes period wise returns for portfolio weighted by factor
    values.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    demeaned : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    group_adjust : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    equal_weight : bool, optional
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation
    by_asset: bool, optional
        If True, returns are reported separately for each esset.
    long_short : str
        when demeand is True, if long_short=='long_short',compute weight both 
        weight>0 and <0; if long_short=='long',compute weight only weight>0; 
        if long_short=='short',compute weight only weight<0
   
    Returns
    -------
    returns : pd.DataFrame
        Period wise factor returns
    """

    weights = \
        factor_weights(factor_data, demeaned, group_adjust, equal_weight,long_short)

    weighted_returns = \
        factor_data[['1D']] \
        .multiply(weights, axis=0)

    if by_asset:
        returns = weighted_returns
    else:
        returns = weighted_returns.groupby(level='date').sum()

    return returns


def cumulative_returns(returns):

    """

    Computes cumulative returns from simple daily returns.
    Parameters
    ----------
    returns: pd.Series

        pd.Series containing daily factor returns (i.e. '1D' returns).
    Returns

    -------
    Cumulative returns series : pd.Series

        Example:

            2015-01-05   1.001310

            2015-01-06   1.000805

            2015-01-07   1.001092

            2015-01-08   0.999200
    """
    return ep.cum_returns(returns, starting_value=1)


def demean_forward_returns(factor_data, grouper=None):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a
    group neutral portfolio constraint and thus allows allows the
    factor to be evaluated across groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period
    return for the Technology stocks in our universe was 0.5% in the
    same period, the group adjusted 5 period return for AAPL in this
    period is -0.4%.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values('date')

    factor_data[['1D']] = factor_data.groupby(grouper)[['1D']] \
        .transform(lambda x: x - x.mean())

    return factor_data


def mean_return_by_quantile(factor_data,
                            by_date=False,
                            by_group=False,
                            demeaned=True,
                            group_adjust=False):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    by_date : bool
        If True, compute quantile bucket returns separately for each date.
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
    demeaned : bool
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level.

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if group_adjust:
        grouper = [factor_data.index.get_level_values('date')] + ['group']
        factor_data = demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    grouper = ['factor_quantile', factor_data.index.get_level_values('date')]

    if by_group:
        grouper.append('group')

    group_stats = factor_data.groupby(grouper)[['1D']] \
        .agg(['mean', 'std', 'count'])

    mean_ret = group_stats.T.xs('mean', level=1).T

    if not by_date:
        grouper = [mean_ret.index.get_level_values('factor_quantile')]
        if by_group:
            grouper.append(mean_ret.index.get_level_values('group'))
        group_stats = mean_ret.groupby(grouper)\
            .agg(['mean', 'std', 'count'])
        mean_ret = group_stats.T.xs('mean', level=1).T

    std_error_ret = group_stats.T.xs('std', level=1).T \
        / np.sqrt(group_stats.T.xs('count', level=1).T)

    return mean_ret, std_error_ret


