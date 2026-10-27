import numpy as np
import pandas as pd
from collections import Iterable
from math import *
from matplotlib import pyplot as plt
from scipy.io import loadmat
from functools import reduce
import datetime
import pickle
from dateutil.relativedelta import relativedelta as timedelta
from stats_tool import *
import pdb

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 泛用函数包

def to_date(d):
    # 将alpha包中的数字日期转化为datetime64
    basis = np.datetime64('-001-12-31')
    return np.array(d) + basis

def nearest(date, daylist, back=1, verbose = False):
    # 寻找指定时间轴上最近（默认往回）的交易日；输入字符串，返回日期
    daylist=np.array(daylist)
    dateline = to_date(daylist) if (isinstance(daylist[0],int) or isinstance(daylist[0],np.int32)) else \
                            pd.to_datetime(daylist).values.astype('datetime64[D]') 
    date = np.datetime64(date)
    
    if back == 0: # 选择最近
        back_line=dateline[dateline<=date]
        for_line=dateline[dateline>=date]
        if len(back_line) == 0:
            back = -1
        elif len(for_line) == 0:
            back = 1
        else:            
            delta = ((for_line[0] - date) -  (date - back_line[-1])).astype(int)
            back = 1 if delta >= 0 else -1
                
    sel=daylist[dateline<=date] if back==1 else daylist[dateline>=date]
    if len(sel)==0:
        if verbose:
            print('Date not found.')
        return daylist[0] if back==1 else daylist[-1]
    else:
        return sel[-1] if back==1 else sel[0]

def month_cal(date, delta=1):
    # 月份加减法，返回datetime64
    res=np.datetime64(pd.to_datetime(date).date()).item()+timedelta(months=delta)
    return np.datetime64(res)

def day_cal(date, delta=1):
    res=np.datetime64(pd.to_datetime(date).date()).item()+timedelta(days=delta)
    return np.datetime64(res)

def date_gener_cs(start, end, start_half=True, end_half=True):
    date_str=lambda x: str(x) if x>=10 else '0{0}'.format(x)
    out=[]
    date_list=[date_str(i+1) for i in range(30)]
    for year in range(start, end+1):
        if year <=2013:
            out+=[str(year)+'-07-01', str(year+1)+'-01-01']
        else:
            for month in ['06', '12']:
                day=['{0}-{1}-'.format(year, month)+j for j in date_list]
                weekday=[datetime.datetime.strptime(j, '%Y-%m-%d').weekday() for j in day]
                index_get=np.argwhere(np.array(weekday)==4)[1,0]+3
                out.append(day[index_get])
                
    if not start_half:
        out=out[1:]
    if not end_half:
        out=out[:-1]
    return out

#%% 市场标签

def mkt_tag(code):
    # 判断市场标签
    
    if code[:2] == '68':
        return 'kcb'
    elif code[:2] == '30':
        return'cyb'
    elif code[0] == '8':
        return 'bjs'
    else:
        return 'zb'