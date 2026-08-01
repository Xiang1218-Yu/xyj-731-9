#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
生成演示数据，用于在没有通达信数据的情况下测试项目
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import user_config as ucfg


def generate_stock_data(stock_code, start_date, end_date, base_price=10):
    """生成单只股票的模拟日线数据"""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
    n = len(dates)
    
    # 生成随机价格数据
    np.random.seed(int(stock_code))
    returns = np.random.normal(0.001, 0.02, n)  # 日收益率
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 生成OHLC数据
    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'code': stock_code,
        'open': prices * (1 + np.random.normal(0, 0.005, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        'close': prices,
        'vol': np.random.randint(1000000, 10000000, n),
        'amount': np.random.randint(10000000, 100000000, n),
    }
    
    df = pd.DataFrame(data)
    
    # 确保 high >= open, close, low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # 添加额外字段
    df['adj'] = 1.0
    df['流通股'] = np.random.randint(100000000, 1000000000, n)
    df['流通市值'] = df['close'] * df['流通股']
    df['换手率'] = df['vol'] / df['流通股'] * 100
    
    return df


def generate_index_data(index_code, start_date, end_date, base_price=3000):
    """生成指数数据"""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(dates)
    
    np.random.seed(int(index_code[-6:]))
    returns = np.random.normal(0.0005, 0.015, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = {
        'date': dates.strftime('%Y-%m-%d'),
        'code': index_code[-6:],
        'open': prices * (1 + np.random.normal(0, 0.003, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.008, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.008, n))),
        'close': prices,
        'vol': np.random.randint(10000000, 100000000, n),
        'amount': np.random.randint(100000000, 1000000000, n),
        'adj': 1.0,
        '流通股': 0,
        '流通市值': 0,
        '换手率': 0,
    }
    
    df = pd.DataFrame(data)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def main():
    print("开始生成演示数据...")
    
    # 创建必要的目录
    for path in [ucfg.tdx['csv_lday'], ucfg.tdx['pickle'], ucfg.tdx['csv_index'], ucfg.tdx['csv_cw']]:
        os.makedirs(path, exist_ok=True)
    
    # 生成股票列表 - 使用rqalpha bundle中存在的真实股票代码
    # 605117和688001等可能不在bundle中，移除
    stock_codes = [
        '000001', '000002', '000063', '000100', '000333',
        '000568', '000651', '000725', '000768', '000858',
        '000895', '002001', '002007', '002024', '002027',
        '002142', '002230', '002236', '002271', '002304',
        '002352', '002415', '002460', '002475', '002594',
        '300001', '300014', '300015', '300033', '300059',
        '300122', '300124', '300274', '300408', '300413',
        '300433', '300498', '300750', '600000', '600009',
        '600016', '600028', '600030', '600031', '600036',
        '600048', '600050', '600104', '600276', '600309',
        '600340', '600406', '600436', '600438', '600519',
        '600547', '600570', '600585', '600588', '600660',
        '600690', '600703', '600745', '600809', '600837',
        '600887', '600900', '601012', '601066', '601088',
        '601100', '601138', '601166', '601211', '601288',
        '601318', '601336', '601398', '601601', '601628',
        '601668', '601688', '601766', '601857', '601888',
        '601899', '601901', '601933', '603288', '603501',
        '603659', '603799', '603986',
    ]
    
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 生成股票数据
    print(f"生成 {len(stock_codes)} 只股票的日线数据...")
    for i, code in enumerate(stock_codes):
        df = generate_stock_data(code, start_date, end_date)
        csv_path = os.path.join(ucfg.tdx['csv_lday'], f'{code}.csv')
        pkl_path = os.path.join(ucfg.tdx['pickle'], f'{code}.pkl')
        df.to_csv(csv_path, index=False, encoding='gbk')
        df.to_pickle(pkl_path)
        if (i + 1) % 20 == 0:
            print(f"  已生成 {i + 1}/{len(stock_codes)} 只股票")
    
    # 生成指数数据
    print("生成指数数据...")
    index_codes = ['999999', '000300', '399001']
    index_names = ['sh999999', 'sh000300', 'sz399001']
    for code, name in zip(index_codes, index_names):
        df = generate_index_data(name, start_date, end_date)
        csv_path = os.path.join(ucfg.tdx['csv_index'], f'{code}.csv')
        df.to_csv(csv_path, index=False, encoding='gbk')
    
    # 生成股本变迁数据
    print("生成股本变迁数据...")
    gbbq_data = {
        'code': ['000001', '000002'],
        '权息日': ['2021-06-15', '2021-07-20'],
        '类别': ['除权除息', '送配股上市'],
        '分红-前流通盘': [1000000000, 2000000000],
        '配股价-前总股本': [0, 0],
        '送转股-后流通盘': [1100000000, 2200000000],
        '配股-后总股本': [1100000000, 2200000000],
    }
    df_gbbq = pd.DataFrame(gbbq_data)
    df_gbbq.to_csv(os.path.join(ucfg.tdx['csv_gbbq'], 'gbbq.csv'), index=False, encoding='gbk')
    
    # 生成财务数据
    print("生成财务数据...")
    cw_data = {
        'code': stock_codes,
        '报告期': ['2023-12-31'] * len(stock_codes),
        '净利润': np.random.randint(100000000, 10000000000, len(stock_codes)),
        '营业收入': np.random.randint(1000000000, 100000000000, len(stock_codes)),
    }
    df_cw = pd.DataFrame(cw_data)
    df_cw.to_pickle(os.path.join(ucfg.tdx['csv_cw'], 'gpcw202312.pkl'))
    
    print("演示数据生成完成！")
    print(f"股票数据: {ucfg.tdx['csv_lday']}")
    print(f"指数数据: {ucfg.tdx['csv_index']}")
    print(f"pickle数据: {ucfg.tdx['pickle']}")


if __name__ == '__main__':
    main()
