import numpy as np
import pandas as pd
from . import data_processing as dp
from . import compute_factor_return as cfr
from . import plotting as pt






def factor_eval_all(alpha_pre, price_data, quantiles=5 ,long_short = 'long_short'):
    """
    以所有股票为样本绘制累计收益图、quantile收益条形图、quantile累计收益图
    
    Parameters
    ----------
    alpha_pre: pd.DataFrame
        从API下载得到的因子数据
    
    price_data: pd.DataFrame
        从API下载得到的股票价格数据
    
    quantiles: int
        分层数量
    
    long_short: str
        when demeand is True, if long_short=='long_short',compute weight both 
        weight>0 and <0; if long_short=='long',compute weight only weight>0; 
        if long_short=='short',compute weight only weight<0
   
    
    """
    factor,prices=dp.data_format(alpha_pre,price_data)
    factor_data=dp.get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=None,
                                         binning_by_group=False,
                                         quantiles=5,
                                         bins=None,
                                         groupby_labels=None,
                                         max_loss=0.35,
                                         zero_aware=False
                                        )
    factor_returns=cfr.factor_returns(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False,
                   by_asset=False,long_short=long_short)
    pt.plot_cumulative_returns(factor_returns, title=None, ax=None)
    mean_ret_by_q,mean_ret_by_std=cfr.mean_return_by_quantile(factor_data,
                            by_date=False,
                            by_group=False,
                            demeaned=True,
                            group_adjust=False)
    pt.plot_quantile_returns_bar(mean_ret_by_q,
                              by_group=False,
                              ylim_percentiles=None,
                              ax=None)
    quantile_returns,quantile_returns_std=cfr.mean_return_by_quantile(factor_data,
                            by_date=True,
                            by_group=False,
                            demeaned=True,
                            group_adjust=False)
    pt.plot_cumulative_returns_by_quantile(quantile_returns,
                                        ax=None)
    

sector_labels = {'801010':'农林牧渔',    
                 '801020':'采掘',    
                 '801030':'化工',    
                 '801040':'钢铁',    
                 '801050':'有色金属',    
                 '801080':'电子',    
                 '801110':'家用电器',    
                 '801120':'食品饮料',    
                 '801130':'纺织服装',    
                 '801140':'轻工制造',    
                 '801150':'医药生物',    
                 '801160':'公用事业',    
                 '801170':'交通运输',    
                 '801180':'房地产',    
                 '801200':'商业贸易',    
                 '801210':'休闲服务',    
                 '801230':'综合',    
                 '801710':'建筑材料',    
                 '801720':'建筑装饰',    
                 '801730':'电气设备',    
                 '801740':'国防军工',    
                 '801750':'计算机',    
                 '801760':'传媒',    
                 '801770':'通信',    
                 '801780':'银行',    
                 '801790':'非银金融',    
                 '801880':'汽车',    
                 '801890':'机械设备',    
                 }
def compute_sector_dict(sector):
    """
    get sector dictionary
    
    Parameters
    ----------
    sector: pd.DataFrame
    从API下载的行业分类数据
    """
    sector_dict = sector.iloc[-1,:] 
    sector_dict.fillna(0,inplace = True) 
    sector_dict = sector_dict.to_dict()
    return sector_dict


def see_sector_name(p):
    #for example: input: '801010'   output: '农林牧渔'
    return sector_labels[p]

def factor_eval_each_sector(alpha_pre, price_data, groupby,by_group=False,quantiles=5, long_short = 'long_short'):
    """
    分行业绘制累计收益图、行业内quantile收益条形图、行业内quantile累计收益图
    最后绘制每个行业的factor weighted平均每日收益条形图
    
    Parameters
    ----------
    alpha_pre: pd.DataFrame
        从API下载得到的因子数据
    
    price_data: pd.DataFrame
        从API下载得到的股票价格数据
    
    groupby: dict
        从compute_sector_dict(sector)得到的申万一级板块分类情况
    
    by_group: bool
        if True, 分行业对因子分层
        else, 按总体对因子分层
    
    quantiles: int
        分层数量
    
    long_short: str
        when demeand is True, if long_short=='long_short',compute weight both 
        weight>0 and <0; if long_short=='long',compute weight only weight>0; 
        if long_short=='short',compute weight only weight<0
   
    
    """
    
    factor,prices=dp.data_format(alpha_pre,price_data)
    factor_data=dp.get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=groupby,
                                         binning_by_group=by_group,
                                         quantiles=5,
                                         bins=None,
                                         filter_zscore=20,
                                         groupby_labels=None,
                                         max_loss=0.35,
                                         zero_aware=False
                                        )
    
    return_each_sector={}
    return_std_each_sector={}
    for p in sector_labels.keys():
        factor_data_p=factor_data[factor_data['group']==p]
        factor_returns=cfr.factor_returns(factor_data_p,
                       demeaned=True,
                       group_adjust=False,
                       equal_weight=False,
                       by_asset=False,
                       long_short=long_short)
        return_each_sector[p]=factor_returns.mean()
        return_std_each_sector[p]=factor_returns.std()
        pt.plot_cumulative_returns(factor_returns, title=p, ax=None)
        mean_ret_by_q,mean_ret_by_std=cfr.mean_return_by_quantile(factor_data_p,
                            by_date=False,
                            by_group=False,
                            demeaned=True,
                            group_adjust=False)
        
        pt.plot_quantile_returns_bar(mean_ret_by_q,title=p,
                                  by_group=False,
                                  ylim_percentiles=None,
                                  ax=None)
        quantile_returns,quantile_returns_std=cfr.mean_return_by_quantile(factor_data_p,
                                by_date=True,
                                by_group=False,
                                demeaned=True,
                                group_adjust=False)
        pt.plot_cumulative_returns_by_quantile(quantile_returns,title=p,
                                            ax=None)
    df = pd.DataFrame.from_dict(return_each_sector)
    df=df.T-df.T.mean()
    pt.plot_sector_returns_bar(df,
                                  by_group=False,
                                  ylim_percentiles=None,
                                  ax=None)
    return df

        
def factor_eval_one_sector(alpha_pre, price_data,p,groupby,by_group=False,quantiles=5, long_short = 'long_short'):
    factor,prices=dp.data_format(alpha_pre,price_data)
    factor_data=dp.get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=groupby,
                                         binning_by_group=by_group,
                                         quantiles=5,
                                         bins=None,
                                         filter_zscore=20,
                                         groupby_labels=None,
                                         max_loss=0.35,
                                         zero_aware=False
                                        )
    
    return_each_sector={}
    return_std_each_sector={}
    
    factor_data_p=factor_data[factor_data['group']==p]
    factor_returns=cfr.factor_returns(factor_data_p,
                       demeaned=True,
                       group_adjust=False,
                       equal_weight=False,
                       by_asset=False,
                       long_short=long_short)
    return_each_sector[p]=factor_returns.mean()
    return_std_each_sector[p]=factor_returns.std()
    pt.plot_cumulative_returns(factor_returns, title=p, ax=None)
    mean_ret_by_q,mean_ret_by_std=cfr.mean_return_by_quantile(factor_data_p,
                            by_date=False,
                            by_group=False,
                            demeaned=True,
                            group_adjust=False)
        
    pt.plot_quantile_returns_bar(mean_ret_by_q,title=p,
                                  by_group=False,
                                  ylim_percentiles=None,
                                  ax=None)
    quantile_returns,quantile_returns_std=cfr.mean_return_by_quantile(factor_data_p,
                                by_date=True,
                                by_group=False,
                                demeaned=True,
                                group_adjust=False)
    pt.plot_cumulative_returns_by_quantile(quantile_returns,title=p,
                                            ax=None)
    df = pd.DataFrame.from_dict(return_each_sector)
    df=df.T-df.T.mean()
    pt.plot_sector_returns_bar(df,
                                  by_group=False,
                                  ylim_percentiles=None,
                                  ax=None)
    return df

    
    
    
    
    
    
    
    
    
    
    

