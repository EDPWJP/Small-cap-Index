
# 导入数据包及处理引擎（自定义包）
from backtest import *
from load_alpha import *

#%% 创建数据引擎
runner = index_backtest(alpha = alpha)


#%% 时间轴设置

date_fix = ['-04-30', '-10-31']
check_time_line = [f'{year}{fix}' for year in range(2014, 2024) for fix in date_fix]
check_time_line = check_time_line[1:]

fix_time_line = date_gener_cs(start = 2014, end = 2023, start_half=False, end_half=True)

#%%

hist_comp = {}

# 遍历数据观察时间轴
for i_date in range(len(check_time_line)):
    
    # 当前观察日
    ob_date = check_time_line[i_date]
    
    # 实际调仓日及向后寻找月份，以提取其他指数成分
    reb_date = fix_time_line[i_date]
    reb_date_near = nearest(reb_date, runner.time_line_m, back = -1)
    
    # 时间处理
    loc_ob = date_order(ob_date, runner.time_line, tag = 1)
    
    ob_start = day_cal(month_cal(ob_date, -12), 1) # 近一年时间
    loc_st = date_order(ob_start, runner.time_line, tag = -1)
    
    loc_ob_m = date_order(ob_date, runner.time_line_m) # 月度索引
    
    # 提取样本空间，与中证全指一致
    # 注意，出于历史回溯原因，这里预先调用了未来中证全指的成分。若实际预测则需要重新构建中证全指样本空间
    order_space = runner.base_comp(date = reb_date_near, base = 'ZZquanzhi')

    # 成交额筛选，以90%为门槛
    
    amt_array = np.nanmean(alpha['dailyinfo_1']['amt'][0][0][order_space, loc_st : loc_ob + 1], axis = 1)  # 提取数据包中观察区间内的成交额数据
    amt_array_fix = pd.Series(amt_array).dropna().sort_values(ascending = False)                           # 剔除NA后排序
    amt_hurdle_count = ceil(len(amt_array) * 0.9)                                                          # 计算90%的对应门槛
    
    amt_hurdle = amt_array_fix.values[amt_hurdle_count]
    amt_filter = (amt_array >= amt_hurdle)
    
    # 剔除总市值前1500的成分股
    # 注意，中证计算市值采用的份额为A股份额（share_totala）
    mkt_cap_window = np.nanmean((alpha['dailyinfo']['close'][0][0][order_space, loc_st : loc_ob + 1] *\
                     alpha['dailyinfo_1']['share_totala'][0][0][order_space, loc_st : loc_ob + 1]), axis = 1)
    
    mkt_cap_hurdle = pd.Series(mkt_cap_window).dropna().sort_values(ascending = False).values[1500]       # 计算1500名的市值门槛
    
    mkt_cap_filter = (mkt_cap_window < mkt_cap_hurdle)
    
    # 同时进行样本空间成交额和总市值的剔除
    order_left = order_space[(amt_filter & mkt_cap_filter)]
    
    # 剔除800月1000的指数样本；同样采用未来信息
    order_800 = runner.base_comp(date = reb_date_near, base = 'ZZ800')
    order_1000 = runner.base_comp(date = reb_date_near, base = 'ZZ1000')
    order_to_drop = set(order_800).union(order_1000)
    
    order_drop_index = np.array(list(set(order_left).difference(order_to_drop)))
    
    # 设置调仓相关参数
    total_num = 2000  # 总样本数
    max_turn = 400    # 最大换手数
    prior_num = 1600  # 优先区数量
    buffer_num = 800  # 缓冲区数量
    
    # 若成分股小于2000则直接纳入（早期情况，不确定是否有特殊处理）
    if len(order_drop_index) <= 2000:
        
        order_final = order_drop_index
    
    else:
        # 计算近一年日均市值
        mkt_cap_order = np.nanmean((alpha['dailyinfo']['close'][0][0][order_drop_index, loc_st : loc_ob + 1] *\
                     alpha['dailyinfo_1']['share_totala'][0][0][order_drop_index, loc_st : loc_ob + 1]), axis = 1)
        
        # 构建数据DataFrame，保证按照市值排序
        order_df = pd.DataFrame({'order':order_drop_index, 'cap':mkt_cap_order}).sort_values(by = 'cap', ascending = False)
        
        # 逐步讨论缓冲机制
        # 若为迭代的第一期，则不考虑缓冲
        if len(hist_comp) == 0:
            order_final = order_df.iloc[:total_num, :]['order'].values
            
        else:
            # 提取旧成分
            last_key = list(hist_comp.keys())[-1]
            last_order = hist_comp[last_key]['order'].values
            
            order_df['last'] = order_df['order'].apply(lambda x: x in last_order)
            
            # 1 处理前1600优先带
            prior_df = order_df.iloc[:prior_num, :]
            buffer_df = order_df.iloc[prior_num : prior_num + buffer_num, :]
            
            prior_new = prior_df[~ prior_df['last']]
            
            # 若优先新成分超过换手上限，则直接取上限
            if len(prior_new) > max_turn:
                
                order_new = prior_new.iloc[:prior_num]['order'].values
                order_last = order_df[order_df['last']].iloc[:total_num - prior_num, :]['order'].values
                
                order_final = np.concatenate((order_new, order_last))
                
            else:
                # 未超上限，则1600优先纳入
                order_prior = prior_df['order'].values
                
                # 剩余成分数
                left_comp = total_num - prior_num
                # 剩余换手数
                left_turn = max_turn - len(prior_new)
                
                buffer_last = buffer_df[buffer_df['last']]
                buffer_new = buffer_df[~buffer_df['last']]
                
                # 缓冲带中老样本是否充足，若充足则全部纳入
                if len(buffer_last) >= left_comp:
                    order_buffer = buffer_last.iloc[:left_comp, :]['order'].values
                
                # 若不充足，则优先纳入全部老样本，随后纳入剩余新样本
                else:
                    order_buffer_last = buffer_last['order'].values
                    left_comp_new = left_comp - len(order_buffer_last)
                    order_buffer_new = buffer_new.iloc[:left_comp_new, :]['order'].values
                    
                    order_buffer = np.concatenate((order_buffer_last, order_buffer_new))
                    
                order_final = np.concatenate((order_prior, order_buffer))
            
    
    stock_list_final = runner.get_stock_code(order_final)
    out_df = pd.DataFrame({'order':order_final, 'code':stock_list_final})
    
    hist_comp[reb_date] = out_df
    print(f'{reb_date} finished. Total num {len(stock_list_final)}')
        
#%% 输出并保存结果

writer = pd.ExcelWriter('中证2000历史成分复现.xlsx')
for key, df in hist_comp.items():
    
    temp_df = df.copy()
    temp_df['name'] = runner.get_stock_name(temp_df['order'].values)
    temp_df = temp_df[['code', 'name']]
    temp_df.columns = ['代码', '简称']
    temp_df.to_excel(writer, sheet_name = key, index = None)
    
writer.close()



