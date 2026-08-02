"""
为日线数据添加全部股票的历史策略买点列。
由于策略需要随时修改调整，因此单独写了策略写入文件，没有整合进readTDX_lday.py

更新内容：
  - 多进程并行处理不同股票区间的策略信号计算
  - 支持 SQLite 数据库持久化存储（字段：股票代码、交易日期、买入/卖出信号、触发策略、前复权价格）
  - strategy 字段动态记录实际触发的策略名称（多因子模式下记录匹配的子策略列表）
  - 支持 --multi 多因子配置、--buy-func/--sell-func 自定义策略函数名
  - 保留原有 CSV 输出，向后兼容

命令行参数：
  del                  完全重新生成策略信号
  single               单进程执行（默认多进程）
  --db [path]          启用 SQLite 存储，可指定数据库路径（默认 strategy_signals.db）
  --no-csv             不输出 celue汇总.csv（默认仍输出）
  --workers N          指定进程数
  --multi <file>       多因子策略 JSON 配置文件
  --buy-func <name>    买入信号函数名（CeLue.py中，默认策略2）
  --sell-func <name>   卖出信号函数名（默认卖策略）
"""
import os
import sys
import time
import sqlite3
from multiprocessing import Pool, RLock, freeze_support
import numpy as np
import pandas as pd
from tqdm import tqdm
from rich import print

import CeLue  # 个人策略文件，不分享
import func
import user_config as ucfg
import strategy_engine

# 变量定义
要剔除的通达信概念 = ["ST板块", ]  # list类型。通达信软件中查看"概念板块"。
要剔除的通达信行业 = ["T1002", ]  # list类型。记事本打开 通达信目录\incon.dat，查看#TDXNHY标签的行业代码。T1002=证券

# 默认数据库路径
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategy_signals.db')


def init_sqlite_db(db_path):
    """
    初始化 SQLite 数据库，创建策略信号表。
    表结构：股票代码、交易日期、买入/卖出信号、触发策略、前复权开高低收价格
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategy_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            celue_buy INTEGER DEFAULT 0,
            celue_sell INTEGER DEFAULT 0,
            strategy TEXT DEFAULT '',
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(code, date)
        )
    ''')
    # 创建索引以加速查询
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_code ON strategy_signals(code)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_date ON strategy_signals(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_buy ON strategy_signals(celue_buy)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_sell ON strategy_signals(celue_sell)')
    conn.commit()
    conn.close()


def save_to_sqlite(db_path, df_signals, batch_size=1000):
    """
    将策略信号 DataFrame 批量写入 SQLite 数据库。
    使用 executemany + 事务批量提交，大幅提升大数据量写入性能。
    利用 INSERT OR REPLACE 实现增量更新。

    :param batch_size: 每批提交的行数，默认1000
    :return: 写入的总行数
    """
    if df_signals is None or len(df_signals) == 0:
        return 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 性能优化：WAL模式 + 正常同步级别，写入速度可提升数倍
    cursor.execute('PRAGMA journal_mode = WAL')
    cursor.execute('PRAGMA synchronous = NORMAL')

    rows_inserted = 0
    try:
        # 预处理所有行为元组列表，避免 iterrows 逐行构造的开销
        all_rows = []
        for _, row in df_signals.iterrows():
            buy_val = 1 if row.get('celue_buy', False) else 0
            sell_val = 1 if row.get('celue_sell', False) else 0

            # 优先从 DataFrame 的 strategy 列读取实际触发策略名称（多因子模式下动态记录）
            # 若该列不存在或为空，则根据买卖信号回退推断策略名
            strategy_str = ''
            if 'strategy' in row.index and pd.notna(row.get('strategy')) and str(row.get('strategy')).strip():
                strategy_str = str(row.get('strategy')).strip()
            else:
                strategy_parts = []
                if buy_val:
                    strategy_parts.append('策略2买入')
                if sell_val:
                    strategy_parts.append('卖策略')
                strategy_str = ','.join(strategy_parts)

            date_val = row['date']
            if hasattr(date_val, 'strftime'):
                date_val = date_val.strftime('%Y-%m-%d')
            else:
                date_val = str(date_val)[:10]

            all_rows.append((
                str(row['code']),
                date_val,
                buy_val,
                sell_val,
                strategy_str,
                float(row['open']) if pd.notna(row.get('open')) else None,
                float(row['high']) if pd.notna(row.get('high')) else None,
                float(row['low']) if pd.notna(row.get('low')) else None,
                float(row['close']) if pd.notna(row.get('close')) else None,
            ))

        # 分批 executemany 写入，避免单次事务过大
        sql = '''
            INSERT OR REPLACE INTO strategy_signals
            (code, date, celue_buy, celue_sell, strategy, open, high, low, close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        for i in range(0, len(all_rows), batch_size):
            batch = all_rows[i:i + batch_size]
            cursor.executemany(sql, batch)
            conn.commit()
            rows_inserted += len(batch)
    except Exception as e:
        conn.rollback()
        print(f'[red]SQLite 写入失败: {e}[/red]')
        raise
    finally:
        conn.close()
    return rows_inserted


# 多进程 worker 的全局引擎变量，通过 initializer 在每个子进程启动时构建一次
_worker_engine = None
_worker_engine_config = None


def _init_celue_worker(engine_config, lock):
    """进程池初始化函数：在每个子进程启动时构建一次多因子策略引擎"""
    global _worker_engine, _worker_engine_config
    tqdm.set_lock(lock)
    _worker_engine_config = engine_config
    if engine_config is not None:
        _worker_engine = strategy_engine.build_engine_from_config(engine_config)


def celue_save(file_list, HS300_信号, tqdm_position=None,
               buy_strategy_name='策略2', sell_strategy_name='卖策略',
               engine_config=None):
    """
    为股票列表计算策略买卖信号并写回pkl/csv。

    :param buy_strategy_name: 买入信号函数名（CeLue.py中的函数名），默认'策略2'
    :param sell_strategy_name: 卖出信号函数名，默认'卖策略'
    :param engine_config: 多因子策略引擎配置dict，为None时使用 buy_strategy_name 指定的单策略
    :return: DataFrame 包含所有触发买卖信号的行，含 strategy 列记录实际触发策略名
    """
    global _worker_engine, _worker_engine_config

    def lambda_update0(x):
        if type(x) == float:
            x = np.nan
        elif x == '0.0':
            x = np.nan
        return x

    # 多因子模式：若 engine_config 已由 initializer 构建则直接复用，否则兜底构建
    if engine_config is not None and _worker_engine is None:
        _worker_engine = strategy_engine.build_engine_from_config(engine_config)
    if engine_config is not None and _worker_engine_config is None:
        _worker_engine_config = engine_config

    # 动态解析买入/卖出策略函数
    buy_func = getattr(CeLue, buy_strategy_name, CeLue.策略2)
    sell_func = getattr(CeLue, sell_strategy_name, CeLue.卖策略)

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
            for col in ['celue_buy', 'celue_sell', 'strategy']:
                if col in df.columns:
                    del df[col]
        df.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并
        # 分别检查 celue_buy / celue_sell / strategy 列是否存在，缺失则插入；已存在则清洗脏数据
        if 'celue_buy' not in df.columns:
            df.insert(df.shape[1], 'celue_buy', np.nan)
        else:
            df['celue_buy'] = (df['celue_buy']
                               .apply(lambda x: lambda_update0(x))
                               .mask(df['celue_buy'] == 'False', False)
                               .mask(df['celue_buy'] == 'True', True)
                               )
        if 'celue_sell' not in df.columns:
            df.insert(df.shape[1], 'celue_sell', np.nan)
        else:
            df['celue_sell'] = (df['celue_sell']
                                .apply(lambda x: lambda_update0(x))
                                .mask(df['celue_sell'] == 'False', False)
                                .mask(df['celue_sell'] == 'True', True)
                                )
        if 'strategy' not in df.columns:
            df.insert(df.shape[1], 'strategy', '')

        if True in df['celue_buy'].isna().to_list():
            start_date = df.index[np.where(df['celue_buy'].isna())[0][0]]
            end_date = df.index[-1]

            if _worker_engine is not None:
                # 多因子模式：使用策略引擎评估，并记录匹配的子策略名称
                # 买入信号序列由引擎中的主买入策略函数生成（保持与回测兼容）
                celue2 = buy_func(df, HS300_信号, start_date=start_date, end_date=end_date)
                celue_sell = sell_func(df, celue2, start_date=start_date, end_date=end_date)
                df.loc[start_date:end_date, 'celue_buy'] = celue2
                df.loc[start_date:end_date, 'celue_sell'] = celue_sell

                # 对每个买入信号为True的日期，用引擎评估匹配的子策略名称
                buy_mask = celue2 == True
                if buy_mask.any():
                    # 引擎评估获取匹配的子策略列表（取最新交易日结果作为该股票的策略标签）
                    engine_result = _worker_engine.evaluate(df, HS300_信号=HS300_信号,
                                                            start_date=start_date, end_date=end_date)
                    matched_names = ','.join(engine_result.matched_strategies) if engine_result.matched_strategies else buy_strategy_name
                    # 对每个买入信号日记录策略名
                    for idx in df.loc[start_date:end_date].index[buy_mask.values]:
                        sell_on_day = bool(df.at[idx, 'celue_sell']) if pd.notna(df.at[idx, 'celue_sell']) else False
                        parts = []
                        if bool(df.at[idx, 'celue_buy']):
                            parts.append(matched_names)
                        if sell_on_day:
                            parts.append(sell_strategy_name)
                        df.at[idx, 'strategy'] = ','.join(parts)
                # 卖出信号日（无买入信号的）也记录卖出策略名
                sell_only_mask = (celue_sell == True) & (~buy_mask)
                for idx in df.loc[start_date:end_date].index[sell_only_mask.values]:
                    df.at[idx, 'strategy'] = sell_strategy_name
            else:
                # 单策略模式：使用配置的函数名作为策略标签
                celue2 = buy_func(df, HS300_信号, start_date=start_date, end_date=end_date)
                celue_sell = sell_func(df, celue2, start_date=start_date, end_date=end_date)
                df.loc[start_date:end_date, 'celue_buy'] = celue2
                df.loc[start_date:end_date, 'celue_sell'] = celue_sell
                # 动态记录每行实际触发的策略名称
                signal_slice = df.loc[start_date:end_date]
                for idx in signal_slice.index:
                    parts = []
                    if bool(signal_slice.at[idx, 'celue_buy']) if pd.notna(signal_slice.at[idx, 'celue_buy']) else False:
                        parts.append(buy_strategy_name)
                    if bool(signal_slice.at[idx, 'celue_sell']) if pd.notna(signal_slice.at[idx, 'celue_sell']) else False:
                        parts.append(sell_strategy_name)
                    df.at[idx, 'strategy'] = ','.join(parts)

            df.reset_index(drop=True, inplace=True)
            df.to_csv(ucfg.tdx['csv_lday'] + os.sep + stockcode + '.csv', index=False, encoding='gbk')
            df.to_pickle(ucfg.tdx['pickle'] + os.sep + stockcode + ".pkl")
        lefttime_tick = int((time.time() - starttime_tick) / (file_list.index(stockcode) + 1)
                            * (len(file_list) - (file_list.index(stockcode) + 1)))

        # 提取celue是true的列，单独保存到一个df，返回这个df
        df_celue = pd.concat([df_celue, df.loc[df['celue_buy'] | df['celue_sell']]])
        # print(f'{process_info} 已用{(time.time() - starttime_tick):.2f}秒 剩余预计{lefttime_tick}秒')
    df_celue['date'] = pd.to_datetime(df_celue['date'], format='%Y-%m-%d')  # 转为时间格式
    df_celue.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并

    return df_celue


def _parse_args(argv):
    """解析命令行参数"""
    use_db = False
    db_path = DEFAULT_DB_PATH
    output_csv = True
    workers = None
    engine_config_path = None
    buy_func_name = '策略2'
    sell_func_name = '卖策略'
    i = 0
    while i < len(argv):
        if argv[i] == '--db':
            use_db = True
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                db_path = argv[i + 1]
                i += 2
            else:
                i += 1
        elif argv[i] == '--no-csv':
            output_csv = False
            i += 1
        elif argv[i] == '--workers' and i + 1 < len(argv):
            workers = int(argv[i + 1])
            i += 2
        elif argv[i] == '--multi' and i + 1 < len(argv):
            engine_config_path = argv[i + 1]
            i += 2
        elif argv[i] == '--buy-func' and i + 1 < len(argv):
            buy_func_name = argv[i + 1]
            i += 2
        elif argv[i] == '--sell-func' and i + 1 < len(argv):
            sell_func_name = argv[i + 1]
            i += 2
        else:
            i += 1
    return use_db, db_path, output_csv, workers, engine_config_path, buy_func_name, sell_func_name


if __name__ == '__main__':
    print(f'附带命令行参数 del 完全重新生成策略信号, 参数 single 单进程执行(默认多进程)')
    print(f'附加参数 --db [path] 启用SQLite存储, --no-csv 不输出CSV, --workers N 指定进程数')
    print(f'          --multi <file> 多因子配置, --buy-func <name> 买入函数名, --sell-func <name> 卖出函数名')

    use_db, db_path, output_csv, workers, engine_config_path, buy_func_name, sell_func_name = _parse_args(sys.argv[1:])

    # 加载多因子策略配置
    engine_config = None
    if engine_config_path:
        import json
        if not os.path.exists(engine_config_path):
            print(f'[red]错误：多因子配置文件 {engine_config_path} 不存在[/red]')
            sys.exit(1)
        with open(engine_config_path, 'r', encoding='utf-8') as f:
            engine_config = json.load(f)
        print(f'多因子策略配置: {engine_config_path}')
        # 多因子配置中可通过 sell_config 指定卖出策略函数名
        if 'sell_config' in engine_config:
            sc = engine_config['sell_config']
            if 'buy_signal_func' in sc:
                buy_func_name = sc['buy_signal_func']
            if 'sell_func' in sc:
                sell_func_name = sc['sell_func']

    print(f'买入策略函数: {buy_func_name} | 卖出策略函数: {sell_func_name}')

    starttime = time.time()
    df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv', index_col=None, encoding='gbk', dtype={'code': str})
    df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')  # 转为时间格式
    df_hs300.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并
    HS300_信号 = CeLue.策略HS300(df_hs300)
    stocklist = [i[:-4] for i in os.listdir(ucfg.tdx['pickle'])]

    if 'del' in sys.argv[1:]:
        print(f'检测到参数 del, 完全重新生成策略信号')
        # 如果启用 SQLite 且 del 模式，清空旧数据
        if use_db and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                conn.execute('DELETE FROM strategy_signals')
                conn.commit()
                conn.close()
                print(f'已清空 SQLite 数据库旧数据: {db_path}')
            except Exception as e:
                print(f'清空数据库失败: {e}')

    # 初始化 SQLite 数据库
    if use_db:
        init_sqlite_db(db_path)
        print(f'SQLite 数据库已就绪: {db_path}')

    # 确定进程数
    if workers is not None:
        t_num = workers
    elif os.cpu_count() > 8:
        t_num = int(os.cpu_count() / 1.5)
    else:
        t_num = max(1, os.cpu_count() - 2)

    # 构造传给 celue_save 的公共关键字参数
    celue_kwargs = {
        'buy_strategy_name': buy_func_name,
        'sell_strategy_name': sell_func_name,
        'engine_config': engine_config,
    }

    if 'single' in sys.argv[1:]:
        print(f'检测到参数 single, 单进程执行')
        df_celue = celue_save(stocklist, HS300_信号, **celue_kwargs)
    else:
        print(f'多进程执行，进程数: {t_num}')
        freeze_support()  # for Windows support
        tqdm.set_lock(RLock())  # for managing output contention
        if engine_config is not None:
            # 多因子模式：通过 initializer 在每个子进程启动时构建一次引擎
            p = Pool(processes=t_num, initializer=_init_celue_worker,
                    initargs=(engine_config, tqdm.get_lock()))
        else:
            p = Pool(processes=t_num, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        pool_result = []  # 存放pool池的返回对象列表
        for i in range(0, t_num):
            div = int(len(stocklist) / t_num)
            mod = len(stocklist) % t_num
            if i + 1 != t_num:
                pool_result.append(
                    p.apply_async(celue_save, args=(stocklist[i * div:(i + 1) * div], HS300_信号, i),
                                  kwds=celue_kwargs))
            else:
                pool_result.append(
                    p.apply_async(celue_save, args=(stocklist[i * div:(i + 1) * div + mod], HS300_信号, i),
                                  kwds=celue_kwargs))

        p.close()
        p.join()

        # 处理celue汇总.csv文件。保存为csv文件，方便查看
        df_celue = pd.DataFrame()
        # 读取pool的返回对象列表。i.get()是读取方法。拼接每个子进程返回的df
        df_list = []
        for i in pool_result:
            df_list.append(i.get())
        if df_list:
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
    if len(df_celue) > 0:
        df_celue = df_celue[~df_celue['code'].isin(kicklist)]

    # 保存到 SQLite 数据库
    if use_db:
        print(f'写入 SQLite 数据库: {db_path}')
        # 准备数据库写入数据（保留前复权价格字段和实际触发策略名称）
        df_db = df_celue.copy()
        if len(df_db) > 0:
            db_cols = ['code', 'date', 'celue_buy', 'celue_sell', 'strategy', 'open', 'high', 'low', 'close']
            available_cols = [c for c in db_cols if c in df_db.columns]
            df_db = df_db[available_cols]
            rows = save_to_sqlite(db_path, df_db)
            print(f'SQLite 写入完成，共 {rows} 条信号记录')
        else:
            print('无信号数据需要写入 SQLite')

    # 保存独立"celue汇总.csv"文件（向后兼容）
    if output_csv:
        print(f'保存独立"celue汇总.csv"文件')
        if len(df_celue) > 0:
            drop_cols = ["open", "high", "low", "vol", "amount", "adj", "流通股", "流通市值", "换手率"]
            existing_drop = [c for c in drop_cols if c in df_celue.columns]
            df_csv = (df_celue
                      .drop(existing_drop, axis=1)
                      .sort_index()
                      .reset_index(drop=True)
                      )
            df_csv.to_csv(ucfg.tdx['csv_gbbq'] + os.sep + 'celue汇总.csv', index=True, encoding='gbk')
        else:
            # 空结果也生成空CSV，避免下游程序报错
            pd.DataFrame(columns=['code', 'date', 'close', 'celue_buy', 'celue_sell', 'strategy']).to_csv(
                ucfg.tdx['csv_gbbq'] + os.sep + 'celue汇总.csv', index=True, encoding='gbk')
        print(f'celue汇总.csv 已保存')

    print(f'用时 {(time.time() - starttime):.2f} 秒, 全部处理完成，程序退出')
