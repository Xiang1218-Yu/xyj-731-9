"""
此为策略文件，基于模板创建。
"""
import numpy as np
import pandas as pd
import talib
import time
import func
from func_TDX import rolling_window, REF, MA, SMA, HHV, LLV, COUNT, EXIST, CROSS, BARSLAST
from rich import print


def 策略HS300(df_hs300, start_date='', end_date=''):
    """
    HS300信号的作用是，当信号是0时，当日不买股票，1时买入。传出
    """
    if start_date == '':
        start_date = df_hs300.index[0]
    if end_date == '':
        end_date = df_hs300.index[-1]
    df_hs300 = df_hs300.loc[start_date:end_date]
    HS300_CLOSE = df_hs300['close']
    HS300_当日涨幅 = (HS300_CLOSE / REF(HS300_CLOSE, 1) - 1) * 100
    HS300_信号 = ~(HS300_当日涨幅 < -1.5) & ~(HS300_当日涨幅 > 1.5)
    return HS300_信号


def 策略1(df, start_date='', end_date='', mode=None):
    """
    策略1 - 快速筛选
    """
    if start_date == '':
        start_date = df.index[0]
    if end_date == '':
        end_date = df.index[-1]
    df = df.loc[start_date:end_date]

    O = df['open']
    H = df['high']
    L = df['low']
    C = df['close']
    if {'换手率'}.issubset(df.columns):
        换手率 = df['换手率']
    else:
        换手率 = 0

    if mode == 'fast':
        if C.shape[0] < 500 or C.iat[-1] < 9:
            return False

        金额万均 = MA(df['amount'] / 10000, 30)
        流通市值亿 = df['流通市值'] / 100000000

        MA5 = MA(C, 5)

        if df['code'][0][0:2] == "68" or df['code'][0][0:2] == "30":
            TJ04_1 = 1.2
        else:
            TJ04_1 = 1.1
        TJ04_2 = ~((C+0.01) >= np.ceil((np.floor(REF(C, 1)*1000*TJ04_1)-4)/10)/100)
        TJ04 = TJ04_2.iat[-1]

        result = TJ04
    else:
        金额万均 = SMA(df['amount'] / 10000, 30)
        流通市值亿 = df['流通市值'] / 100000000
        MA5 = SMA(C, 5)

        TJ01 = (BARSLAST(C == 0) > 500) & (df['close'] > 9)

        if df['code'][0][0:2] == "68" or df['code'][0][0:2] == "30":
            TJ04_1 = 1.2
        else:
            TJ04_1 = 1.1
        TJ04_2 = ~((C+0.01) >= np.ceil((np.floor(REF(C, 1)*1000*TJ04_1)-4)/10)/100)
        TJ04 = TJ04_2

        result = TJ01 & TJ04
    return result


def 策略2(df, HS300_信号, start_date='', end_date=''):
    """
    策略2 - 详细筛选
    """
    if start_date == '':
        start_date = df.index[0]
    if end_date == '':
        end_date = df.index[-1]
    df = df.loc[start_date:end_date]

    if df.shape[0] < 251:
        return pd.Series(index=df.index, dtype=bool)

    HS300_信号 = pd.Series(HS300_信号, index=df.index, dtype=bool).dropna()

    O = df['open']
    H = df['high']
    L = df['low']
    C = df['close']
    换手率 = df['换手率']

    MA5 = SMA(C, 5)
    MA10 = SMA(C, 10)
    MA20 = SMA(C, 20)
    MA60 = SMA(C, 60)
    MA120 = SMA(C, 120)
    MA250 = SMA(C, 250)

    流通市值亿 = df['流通市值'] / 100000000

    TJ01 = (MA120 > -5) & (MA10 < 60) & (MA60 < 10) & (-7 < MA250) & (MA250 < 10)

    TJ02 = (C > SMA(C, 60)) & (C < SMA(C, 60) * 1.1) & (C > O)

    TJ06_1 = LLV(C, 200)
    TJ06_2 = LLV(C, 20)
    TJ06_MA60_DAY = BARSLAST((REF(C, 5) < MA60) & CROSS(C, MA60))
    TJ06_MA60 = pd.Series(index=TJ06_MA60_DAY.index, dtype=float)

    for i, (k, v) in enumerate(TJ06_MA60_DAY.items()):
        if i - int(v) > 0:
            TJ06_MA60.iloc[i] = MA60.iloc[i - int(v)]

    df = pd.concat([df, TJ06_MA60_DAY.rename('TJ06_MA60_DAY')], axis=1)
    df.insert(df.shape[1], 'TJ06_MA60_LLV', np.nan)
    for index_date in df.loc[df['TJ06_MA60_DAY'] == 0].index.to_list():
        index_int = df.index.get_loc(index_date)
        df.at[index_date, 'TJ06_MA60_LLV'] = df.iloc[index_int - 20:index_int]['close'].min()
    df = df.ffill()  # 向下填充无效值
    TJ06_MA60_LLV = df['TJ06_MA60_LLV']

    TJ06_3 = TJ06_MA60 / TJ06_MA60_LLV
    TJ06_4 = C / TJ06_MA60
    TJ06 = (TJ06_2 / TJ06_1 - 1 < 0.5) & (1 < TJ06_3 / TJ06_4) & (TJ06_3 / TJ06_4 < 1.5)

    TJP1 = 策略1(df, start_date, end_date)
    TJ11_1 = HS300_信号 & TJP1 & TJ01 & TJ02 & TJ06
    TJ11_2 = COUNT(TJ11_1, 10)
    TJ11 = TJ11_1 & (REF(TJ11_2, 1) == 0)

    TJ99 = TJ11

    BUYSIGN = TJ99

    return BUYSIGN


def 卖策略(df, 策略2, start_date='', end_date=''):
    """
    卖出策略
    """
    if True not in 策略2.to_list():
        return pd.Series(index=策略2.index, dtype=bool)

    if start_date == '':
        start_date = df.index[0]
    if end_date == '':
        end_date = df.index[-1]
    df = df.loc[start_date:end_date]

    O = df['open']
    H = df['high']
    L = df['low']
    C = df['close']
    流通市值亿 = df['流通市值'] / 100000000

    MA10 = SMA(C, 10)
    MA60 = SMA(C, 60)

    BUY_TODAY = BARSLAST(策略2)
    BUY_PRICE_CLOSE = pd.Series(index=C.index, dtype=float)
    BUY_PRICE_OPEN = pd.Series(index=C.index, dtype=float)
    BUY_PCT = pd.Series(index=C.index, dtype=float)
    BUY_PCT_MAX = pd.Series(index=C.index, dtype=float)

    for i in BUY_TODAY[BUY_TODAY == 0].index.to_list()[::-1]:
        BUY_PRICE_CLOSE.loc[i] = C.loc[i]
        BUY_PRICE_OPEN.loc[i] = O.loc[i]
        BUY_PRICE_CLOSE.ffill(inplace=True)  # 向下填充无效值
        BUY_PRICE_OPEN.ffill(inplace=True)  # 向下填充无效值
        BUY_PCT = C / BUY_PRICE_CLOSE - 1
        for k, v in BUY_PCT[i:].items():
            if np.isnan(BUY_PCT_MAX[k]):
                BUY_PCT_MAX[k] = BUY_PCT[i:k].max()

    SELL01 = (C < MA60) & (C < BUY_PRICE_OPEN)

    SELL02 = (BUY_PCT < 0.1) & (H < REF(L, 1))

    SELL03_1 = pd.Series(index=C.index, dtype=float)
    SELL03_1 = 流通市值亿.apply(lambda x: 7 if x < 100 else 14)
    SELL03 = (BUY_TODAY > SELL03_1) & (0.01 < C / BUY_PCT) & (C / BUY_PCT < 0.03)

    SELLSIGN01 = SELL01 | SELL02 | SELL03
    SELLSIGN = pd.Series(index=C.index, dtype=bool)

    for i in BUY_TODAY[BUY_TODAY == 0].index.to_list()[::-1]:
        for k, v in SELLSIGN01[i:].items():
            if k != i and SELLSIGN01[k]:
                SELLSIGN[k] = True
                break

    return SELLSIGN
