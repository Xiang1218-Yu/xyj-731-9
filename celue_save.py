"""
为日线数据添加全部股票的历史策略买点列。
由于策略需要随时修改调整，因此单独写了策略写入文件，没有整合进readTDX_lday.py

回测结果持久化：默认同时生成 celue汇总.csv 和 SQLite数据库 celue.db（表celue_signals，
包含股票代码、交易日期、买入/卖出信号、触发策略、前复权价格等字段）。
命令行参数：
    del      完全重新生成策略信号
    single   单进程执行(默认多进程并行处理不同股票区间)
    multi    触发策略列使用多因子引擎逐日计算各子策略命中情况
    nocsv    不生成 celue汇总.csv
    nosqlite 不写入 SQLite 数据库
"""
import os
import sqlite3
import sys
import time
from multiprocessing import Pool, RLock, freeze_support
import numpy as np
import pandas as pd
from tqdm import tqdm
from rich import print

import CeLue  # 个人策略文件，不分享
import celue_engine  # 多因子选股策略引擎
import func
import user_config as ucfg

# 配置部分

start_date = ''
end_date = ''

# 变量定义
要剔除的通达信概念 = ["ST板块", ]  # list类型。通达信软件中查看“概念板块”。
要剔除的通达信行业 = ["T1002", ]  # list类型。记事本打开 通达信目录\incon.dat，查看#TDXNHY标签的行业代码。T1002=证券

# SQLite数据库路径（与celue汇总.csv同目录）
db_path = ucfg.tdx['csv_gbbq'] + os.sep + 'celue.db'

# 回测信号表结构：股票代码、交易日期、买入/卖出信号、触发策略、前复权价格
SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS celue_signals (
    code TEXT NOT NULL,            -- 股票代码
    date TEXT NOT NULL,            -- 交易日期 YYYY-MM-DD
    celue_buy INTEGER DEFAULT 0,   -- 买入信号 1=触发 0=未触发
    celue_sell INTEGER DEFAULT 0,  -- 卖出信号 1=触发 0=未触发
    trigger_strategy TEXT,         -- 触发策略（多因子时为命中的子策略列表，|分隔）
    open REAL,                     -- 前复权开盘价
    high REAL,                     -- 前复权最高价
    low REAL,                      -- 前复权最低价
    close REAL,                    -- 前复权收盘价
    PRIMARY KEY (code, date)
)
"""


def split_stocklist(stocklist, t_num):
    """将股票列表平均切分为t_num个区间，供多进程并行处理。返回区间列表"""
    chunks = []
    div = int(len(stocklist) / t_num)
    mod = len(stocklist) % t_num
    for i in range(0, t_num):
        if i + 1 != t_num:
            chunks.append(stocklist[i * div:(i + 1) * div])
        else:
            chunks.append(stocklist[i * div:(i + 1) * div + mod])
    return chunks


def make_trigger_column(df, HS300_信号):
    """
    生成trigger_strategy列（触发策略）。
    默认单策略模式：买入=策略2，卖出=卖策略；
    命令行参数 multi：使用多因子引擎逐日计算各启用因子的命中情况，|分隔拼接
    """
    if 'multi' in sys.argv[1:]:
        engine = celue_engine.MultiFactorEngine()
        # 策略上下文与xuangu.py的run_multi保持一致，含HS300_信号/start_date/end_date
        context = {'HS300_信号': HS300_信号, 'start_date': start_date, 'end_date': end_date,
                   '_cache': {}}
        trigger = pd.Series('', index=df.index, dtype=object)
        for f in engine.factors:
            name = ('NOT ' if f.get('not', False) else '') + f['name']
            series = engine.factor_series(f['name'], df, context)
            trigger = trigger + np.where(series, name + '|', '')
        return trigger.str.rstrip('|')
    # 单策略模式
    buy_mask = df['celue_buy'] == True
    sell_mask = df['celue_sell'] == True
    trigger = pd.Series('', index=df.index, dtype=object)
    trigger = trigger.mask(buy_mask & ~sell_mask, '策略2')
    trigger = trigger.mask(~buy_mask & sell_mask, '卖策略')
    trigger = trigger.mask(buy_mask & sell_mask, '策略2|卖策略')
    return trigger


def save_to_sqlite(df_celue, path):
    """
    将回测信号结果写入SQLite数据库（全量刷新，与celue汇总.csv的覆盖语义一致）
    :param df_celue: 含 code/date/celue_buy/celue_sell/trigger_strategy/前复权价格列 的DF
    :param path: SQLite数据库文件路径
    """
    df_db = df_celue[['code', 'date', 'celue_buy', 'celue_sell', 'trigger_strategy',
                      'open', 'high', 'low', 'close']].copy()
    df_db['date'] = df_db['date'].dt.strftime('%Y-%m-%d')  # 日期存为字符串
    df_db['celue_buy'] = df_db['celue_buy'].astype(bool).astype(int)  # 布尔转0/1
    df_db['celue_sell'] = df_db['celue_sell'].astype(bool).astype(int)
    conn = sqlite3.connect(path)
    try:
        conn.execute(SQL_CREATE_TABLE)
        conn.execute('DELETE FROM celue_signals')  # 全量刷新
        df_db.to_sql('celue_signals', conn, if_exists='append', index=False)
        conn.commit()
    finally:
        conn.close()
    print(f'已写入SQLite数据库 {path} (表celue_signals，共 {len(df_db)} 行)')


def read_celue_db(path):
    """
    从SQLite数据库读取回测信号（huice.py回测使用）。返回与celue汇总.csv相同结构的DF
    """
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql('SELECT * FROM celue_signals', conn, dtype={'code': str})
    finally:
        conn.close()
    df['celue_buy'] = df['celue_buy'].astype(bool)
    df['celue_sell'] = df['celue_sell'].astype(bool)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  # 转为时间格式
    df.set_index('date', drop=False, inplace=True)  # 时间为索引
    return df


def celue_save(file_list, HS300_信号, tqdm_position=None):
    def lambda_update0(x):
        if type(x) == float:
            x = np.nan
        elif x == '0.0':
            x = np.nan
        return x

    # print('\nRun task (%s)' % os.getpid())
    starttime_tick = time.time()
    df_celue = pd.DataFrame()
    if 'single' in sys.argv[1:]:
        tq = tqdm(file_list)
    else:
        tq = tqdm(file_list, leave=False, position=tqdm_position)
    for stockcode in tq:
        tq.set_description(stockcode)
        # process_info = f'[{(stocklist.index(stockcode) + 1):>4}/{str(len(stocklist))}] {stockcode}'
        pklfile = ucfg.tdx['pickle'] + os.sep + stockcode + ".pkl"
        df = pd.read_pickle(pklfile)
        if 'del' in sys.argv[1:]:
            if 'celue_buy' in df.columns:
                del df['celue_buy']
            if 'celue_sell' in df.columns:
                del df['celue_sell']
        df.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并
        if not {'celue_buy', 'celue_buy'}.issubset(df.columns):
            df.insert(df.shape[1], 'celue_buy', np.nan)  # 插入celue_buy列，赋值NaN
            df.insert(df.shape[1], 'celue_sell', np.nan)  # 插入celue_sell列，赋值NaN
        else:
            # 由于make_fq时fillna将最新的空的celue单元格也填充为0，所以先恢复nan
            df['celue_buy'] = (df['celue_buy']
                               .apply(lambda x: lambda_update0(x))
                               .mask(df['celue_buy'] == 'False', False)
                               .mask(df['celue_buy'] == 'True', True)
                               )

            df['celue_sell'] = (df['celue_sell']
                                .apply(lambda x: lambda_update0(x))
                                .mask(df['celue_sell'] == 'False', False)
                                .mask(df['celue_sell'] == 'True', True)
                                )

        if True in df['celue_buy'].isna().to_list():
            start_date = df.index[np.where(df['celue_buy'].isna())[0][0]]
            end_date = df.index[-1]
            celue2 = CeLue.策略2(df, HS300_信号, start_date=start_date, end_date=end_date)
            celue_sell = CeLue.卖策略(df, celue2, start_date=start_date, end_date=end_date)
            df.loc[start_date:end_date, 'celue_buy'] = celue2
            df.loc[start_date:end_date, 'celue_sell'] = celue_sell
            df.reset_index(drop=True, inplace=True)
            df.to_csv(ucfg.tdx['csv_lday'] + os.sep + stockcode + '.csv', index=False, encoding='gbk')
            df.to_pickle(ucfg.tdx['pickle'] + os.sep + stockcode + ".pkl")
        # 生成触发策略列（multi参数时使用多因子引擎逐日计算子策略命中情况）
        df['trigger_strategy'] = make_trigger_column(df, HS300_信号)
        lefttime_tick = int((time.time() - starttime_tick) / (file_list.index(stockcode) + 1)
                            * (len(file_list) - (file_list.index(stockcode) + 1)))

        # 提取celue是true的列，单独保存到一个df，返回这个df
        信号行 = (df['celue_buy'] == True) | (df['celue_sell'] == True)
        df_celue = pd.concat([df_celue, df.loc[信号行.fillna(False)]])
        # print(f'{process_info} 已用{(time.time() - starttime_tick):.2f}秒 剩余预计{lefttime_tick}秒')
    df_celue['date'] = pd.to_datetime(df_celue['date'], format='%Y-%m-%d')  # 转为时间格式
    df_celue.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并

    return df_celue


if __name__ == '__main__':
    print(f'附带命令行参数 del 完全重新生成策略信号, 参数 single 单进程执行(默认多进程)')
    print(f'参数 multi 触发策略列使用多因子引擎计算, nocsv 不生成csv, nosqlite 不写SQLite数据库')
    starttime = time.time()
    df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv', index_col=None, encoding='gbk', dtype={'code': str})
    df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')  # 转为时间格式
    df_hs300.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并
    HS300_信号 = CeLue.策略HS300(df_hs300)
    stocklist = [i[:-4] for i in os.listdir(ucfg.tdx['pickle'])]

    if 'del' in sys.argv[1:]:
        print(f'检测到参数 del, 完全重新生成策略信号')

    if 'single' in sys.argv[1:]:
        print(f'检测到参数 single, 单进程执行')
        df_celue = celue_save(stocklist, HS300_信号)
    else:
        # 多线程。好像没啥效果提升
        # threads = []
        # t_num = 4  # 线程数
        # for i in range(0, t_num):
        #     div = int(len(stocklist) / t_num)
        #     mod = len(stocklist) % t_num
        #     if i+1 != t_num:
        #         # print(i, i * div, (i + 1) * div)
        #         threads.append(threading.Thread(target=celue_save, args=(stocklist[i*div:(i+1)*div], HS300_信号)))
        #     else:
        #         # print(i, i * div, (i + 1) * div + mod)
        #         threads.append(threading.Thread(target=celue_save, args=(stocklist[i*div:(i+1)*div+mod], HS300_信号)))
        # # celue_save(stocklist, HS300_信号)
        #
        # print(threads)
        # for t in threads:
        #     t.setDaemon(True)
        #     t.start()
        #
        # for t in threads:
        #     t.join()
        # print("\n")

        # 多进程
        # print('Parent process %s' % os.getpid())
        # 进程数 读取CPU逻辑处理器个数
        if os.cpu_count() > 8:
            t_num = int(os.cpu_count() / 1.5)
        else:
            t_num = os.cpu_count() - 2
        freeze_support()  # for Windows support
        tqdm.set_lock(RLock())  # for managing output contention
        p = Pool(processes=t_num, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        pool_result = []  # 存放pool池的返回对象列表
        # 按股票区间切分，每个子进程并行计算一个区间的策略信号
        for i, chunk in enumerate(split_stocklist(stocklist, t_num)):
            pool_result.append(p.apply_async(celue_save, args=(chunk, HS300_信号, i)))
        # celue_save(stocklist, HS300_信号)

        # print('Waiting for all subprocesses done...')
        p.close()
        p.join()

        # 处理celue汇总.csv文件。保存为csv文件，方便查看
        df_celue = pd.DataFrame()
        # 读取pool的返回对象列表。i.get()是读取方法。拼接每个子进程返回的df
        df_list = []
        for i in pool_result:
            df_list.append(i.get())
        df_celue = pd.concat(df_list)

    # df_celue 是处理后的所有股票策略信号汇总文件。
    # 下面处理自定义股票板块剔除

    # 生成要剔除的股票列表 kicklist
    print(f'生成股票列表, 共 {len(stocklist)} 只股票')
    print(f'剔除通达信概念股票: {要剔除的通达信概念}')
    kicklist = []
    try:
        df = func.get_TDX_blockfilecontent("block_gn.dat")
        for i in 要剔除的通达信概念:
            kicklist = kicklist + df.loc[df['blockname'] == i]['code'].tolist()
    except FileNotFoundError:
        print("  通达信板块文件不存在，跳过概念剔除")
    print(f'剔除通达信行业股票: {要剔除的通达信行业}')
    try:
        df = pd.read_csv(ucfg.tdx['tdx_path'] + os.sep + 'T0002' + os.sep + 'hq_cache' + os.sep + "tdxhy.cfg",
                         sep='|', header=None, dtype='object')
        for i in 要剔除的通达信行业:
            kicklist = kicklist + df.loc[df[2] == i][1].tolist()
    except FileNotFoundError:
        print("  通达信行业文件不存在，跳过行业剔除")
    print("剔除科创板股票")
    try:
        tdx_stocks = pd.read_csv(ucfg.tdx['tdx_path'] + '/T0002/hq_cache/infoharbor_ex.code',
                                 sep='|', header=None, index_col=None, encoding='gbk', dtype={0: str})
        kicklist = kicklist + tdx_stocks[0][tdx_stocks[0].apply(lambda x: x[0:2] == "68")].to_list()
    except FileNotFoundError:
        print("  通达信股票列表文件不存在，跳过科创板剔除")
    stocklist = list(filter(lambda i: i not in kicklist, stocklist))
    print(f'共 {len(stocklist)} 只候选股票')
    # df_celue 剔除在kicklist中的股票
    df_celue = df_celue[~df_celue['code'].isin(kicklist)]
    df_celue = df_celue.sort_index()

    # 写入SQLite数据库（回测结果持久化，含触发策略、前复权价格字段）。参数 nosqlite 跳过
    if 'nosqlite' not in sys.argv[1:]:
        save_to_sqlite(df_celue, db_path)

    # 参数 nocsv 跳过csv生成
    if 'nocsv' not in sys.argv[1:]:
        print(f'保存独立"celue汇总.csv"文件')
        df_celue_csv = (df_celue
                        .drop(["open", "high", "low", "vol", "amount", "adj", "流通股", "流通市值", "换手率"], axis=1)
                        .reset_index(drop=True)
                        )
        df_celue_csv.to_csv(ucfg.tdx['csv_gbbq'] + os.sep + 'celue汇总.csv', index=True, encoding='gbk')
    print(f'用时 {(time.time() - starttime):.2f} 秒, 全部处理完成，程序退出')
