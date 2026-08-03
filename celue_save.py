"""
为日线数据添加全部股票的历史策略买点列。
由于策略需要随时修改调整，因此单独写了策略写入文件，没有整合进readTDX_lday.py

本次优化：
1. 多进程并行处理不同股票区间的策略信号（原有能力保留并修复索引跳变问题）。
2. 将回测结果从 CSV 升级为 SQLite 数据库存储（celue.db），
   表 signals 字段包括：股票代码、交易日期、买入信号、卖出信号、触发策略、前复权价格(open/high/low/close)、评分。
   同时保留原 celue汇总.csv 输出，兼容已有回测流程。
3. 新增 --multi-factor 参数，支持通过 JSON 配置多因子组合策略；
   触发策略列(celue_strategies)会记录命中的子策略名称。
4. 新增 --db 参数自定义 SQLite 路径；--no-csv 可关闭 CSV 输出。

兼容旧版命令行参数：del（完全重新生成）、single（单进程执行）。
"""
import os
import sys
import time
import argparse
import sqlite3
from multiprocessing import Pool, RLock, freeze_support
import numpy as np
import pandas as pd
from tqdm import tqdm
from rich import print

# CeLue 依赖 talib，采用延迟导入，使本模块的 SQLite 工具函数可在无 talib 环境下独立使用
CeLue = None


def _ensure_celue():
    global CeLue
    if CeLue is None:
        import CeLue as _CeLue
        CeLue = _CeLue
    return CeLue


import func
import user_config as ucfg
from strategy_engine import StrategyEngine, load_engine_from_json, CompositeStrategy, Strategy

# 变量定义
要剔除的通达信概念 = ["ST板块", ]  # list类型。通达信软件中查看“概念板块”。
要剔除的通达信行业 = ["T1002", ]  # list类型。记事本打开 通达信目录\incon.dat，查看#TDXNHY标签的行业代码。T1002=证券


def _align_bool(series, index):
    """
    将信号 Series 对齐到指定索引并转为 bool 类型。
    - 用 reindex 处理索引类型/精度不一致或缺失日期的情况；
    - 缺失位置统一填充为 False；
    - 先 astype(bool) 再 fillna，避免 pandas 对 object 列做 downcast 时的 FutureWarning。
    """
    aligned = series.reindex(index)
    return aligned.where(aligned.notna(), False).astype(bool)


def parse_args():
    parser = argparse.ArgumentParser(
        description='为日线数据生成策略信号并持久化到 SQLite/CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('legacy_args', nargs='*', help='兼容旧版位置参数：del / single')
    parser.add_argument('--single', action='store_true', help='单进程执行（默认多进程）')
    parser.add_argument('--del', dest='force_del', action='store_true',
                        help='完全重新生成策略信号（删除已有的 celue_buy/celue_sell 列）')
    parser.add_argument('--multi-factor', dest='multi_factor', default=None,
                        help='多因子策略 JSON 配置文件路径（可选，默认使用 策略2 + 卖策略）')
    parser.add_argument('--db', default=None,
                        help='SQLite 数据库路径，默认存放在 csv_gbbq 目录下 celue.db')
    parser.add_argument('--no-csv', dest='no_csv', action='store_true',
                        help='不生成 celue汇总.csv（默认同时生成）')
    parser.add_argument('--workers', type=int, default=0,
                        help='多进程工作进程数，0 表示自动按 CPU 核心数计算')
    args = parser.parse_args()
    # 兼容旧版位置参数
    if 'del' in args.legacy_args:
        args.force_del = True
    if 'single' in args.legacy_args:
        args.single = True
    return args


# ---------------------------------------------------------------------- #
# SQLite 持久化
# ---------------------------------------------------------------------- #
def get_db_path(custom_path=None):
    """返回 SQLite 数据库路径。"""
    if custom_path:
        return custom_path
    return ucfg.tdx['csv_gbbq'] + os.sep + 'celue.db'


def init_db(db_path):
    """初始化 signals 表。"""
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            code          TEXT    NOT NULL,
            trade_date    TEXT    NOT NULL,
            celue_buy     INTEGER NOT NULL,
            celue_sell    INTEGER NOT NULL,
            strategies    TEXT,
            score         REAL,
            open          REAL,
            high          REAL,
            low           REAL,
            close         REAL,
            amount        REAL,
            PRIMARY KEY (code, trade_date)
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(trade_date)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_buy ON signals(celue_buy)')
    conn.commit()
    conn.close()


def save_df_to_db(db_path, df_signals, batch_size=5000):
    """
    把策略信号 DataFrame 批量写入 SQLite（REPLACE 避免主键冲突）。

    使用向量化的列数组 + executemany 批量提交，替代逐行 iterrows 单条插入，
    在大批量信号（数万行）场景下性能提升显著。
    """
    if df_signals is None or len(df_signals) == 0:
        return

    # 统一类型并构造批量元组列表（避免 iterrows 的 Python 级逐行开销）
    df = df_signals.copy()
    df['trade_date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['code'] = df['code'].astype(str)
    df['celue_buy'] = df['celue_buy'].astype(bool).astype(int)
    df['celue_sell'] = df['celue_sell'].astype(bool).astype(int)

    # strategies 列可能是 list/tuple 或逗号字符串，统一转为字符串
    def _fmt_strategies(v):
        if isinstance(v, (list, tuple)):
            return ','.join(str(x) for x in v)
        if pd.isna(v):
            return ''
        return str(v)

    if 'celue_strategies' in df.columns:
        df['strategies'] = df['celue_strategies'].apply(_fmt_strategies)
    else:
        df['strategies'] = ''

    # score 列空值转 None
    if 'celue_score' in df.columns:
        df['score'] = pd.to_numeric(df['celue_score'], errors='coerce')
    else:
        df['score'] = np.nan

    cols = ['code', 'trade_date', 'celue_buy', 'celue_sell', 'strategies', 'score',
            'open', 'high', 'low', 'close', 'amount']
    for c in ['open', 'high', 'low', 'close', 'amount']:
        if c not in df.columns:
            df[c] = np.nan

    # 利用 numpy 数组一次性转成 list[tuple]，比 iterrows 快数倍
    records = list(zip(*[df[c].where(pd.notna(df[c]), None).tolist() for c in cols]))

    sql = (
        'INSERT OR REPLACE INTO signals '
        '(code, trade_date, celue_buy, celue_sell, strategies, score, open, high, low, close, amount) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
    )
    conn = sqlite3.connect(db_path)
    try:
        # 分批 executemany，避免单次事务过大
        for start in range(0, len(records), batch_size):
            conn.executemany(sql, records[start:start + batch_size])
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------- #
# 单只股票策略计算
# ---------------------------------------------------------------------- #
def _engine_to_dict(node):
    """把策略树节点递归转为可 JSON 序列化的 dict（用于多进程传递配置）。"""
    if isinstance(node, CompositeStrategy):
        return {
            'name': node.name, 'op': node.op, 'weight': node.weight,
            'children': [_engine_to_dict(c) for c in node.children],
        }
    elif isinstance(node, Strategy):
        func_ref = node._func
        func_name = func_ref.__name__ if (callable(func_ref) and hasattr(func_ref, '__name__')) else str(node._func)
        return {
            'name': node.name, 'func': func_name, 'weight': node.weight,
            'args': node.strategy_args, 'kwargs': node.strategy_kwargs,
            'needs_hs300': node.needs_hs300, 'is_sell': node.is_sell,
        }
    return {}


def _default_engine_config():
    """默认策略配置（与旧版 celue_save 行为一致）：策略2 作为买入，卖策略作为卖出。"""
    root = CompositeStrategy('默认回测策略', op='AND', children=[
        Strategy('策略2买入', '策略2', weight=1.0, needs_hs300=True),
    ])
    return _engine_to_dict(root)


def compute_stock_signals(stockcode, HS300_信号, force_del=False, engine_config=None):
    """
    计算单只股票的策略信号，并写回其 pkl/csv 文件。
    返回该股票触发了买卖信号的行组成的 DataFrame（包含 celue_strategies / celue_score 列）。
    """
    def lambda_update0(x):
        if type(x) == float:
            x = np.nan
        elif x == '0.0':
            x = np.nan
        return x

    pklfile = ucfg.tdx['pickle'] + os.sep + stockcode + ".pkl"
    df = pd.read_pickle(pklfile)
    if force_del:
        if 'celue_buy' in df.columns:
            del df['celue_buy']
        if 'celue_sell' in df.columns:
            del df['celue_sell']
    df.set_index('date', drop=False, inplace=True)

    if not {'celue_buy', 'celue_sell'}.issubset(df.columns):
        df.insert(df.shape[1], 'celue_buy', np.nan)
        df.insert(df.shape[1], 'celue_sell', np.nan)
    else:
        # 由于 make_fq 时 fillna 将最新空 celue 单元格填充为 0，先恢复 nan
        df['celue_buy'] = (df['celue_buy']
                           .apply(lambda_update0)
                           .mask(df['celue_buy'] == 'False', False)
                           .mask(df['celue_buy'] == 'True', True))
        df['celue_sell'] = (df['celue_sell']
                            .apply(lambda_update0)
                            .mask(df['celue_sell'] == 'False', False)
                            .mask(df['celue_sell'] == 'True', True))

    if 'celue_strategies' not in df.columns:
        df.insert(df.shape[1], 'celue_strategies', '')
    if 'celue_score' not in df.columns:
        df.insert(df.shape[1], 'celue_score', np.nan)

    # 只计算尚未有信号的区间（增量更新）
    if True in df['celue_buy'].isna().to_list():
        start_date = df.index[np.where(df['celue_buy'].isna())[0][0]]
        end_date = df.index[-1]
        df_slice = df.loc[start_date:end_date]

        if engine_config is not None:
            engine = StrategyEngine.from_dict(engine_config)
            result = engine.evaluate_stock_series(df_slice, hs300_signal=HS300_信号,
                                                  start_date=start_date, end_date=end_date)
            # 统一以 df_slice.index 为准对齐，避免不同来源 Series 的索引类型/精度不一致
            # （如 Timestamp 与 datetime64、或引擎内部重建索引）导致按标签 .get() 查找失败
            slice_index = df_slice.index
            buy_signal = _align_bool(result['buy_signal'], slice_index)
            score = result['score'].reindex(slice_index)
            strategies = result['matched_strategies']
            strat_str = ','.join(strategies)
            # 卖出信号：若配置中含 is_sell 策略，则从引擎结果中识别；否则用 卖策略
            celue_mod = _ensure_celue()
            sell_names = engine.get_sell_strategy_names()
            if sell_names:
                # 卖出信号由组合信号中命中的卖出策略决定；这里用独立方式再算一次卖策略
                celue_sell = celue_mod.卖策略(df_slice, buy_signal,
                                         start_date=start_date, end_date=end_date)
            else:
                celue_sell = celue_mod.卖策略(df_slice, buy_signal,
                                         start_date=start_date, end_date=end_date)
            celue_sell = _align_bool(celue_sell, slice_index)

            df.loc[start_date:end_date, 'celue_buy'] = buy_signal.values
            df.loc[start_date:end_date, 'celue_sell'] = celue_sell.values
            # 向量化生成触发策略列：买入信号为 True 的行写入命中策略名，否则为空串
            # 使用 np.where + .values，避免 index.map 内对 Timestamp 标签做 .get() 查找
            df.loc[start_date:end_date, 'celue_strategies'] = np.where(
                buy_signal.values, strat_str, ''
            )
            df.loc[start_date:end_date, 'celue_score'] = score.values
        else:
            # 旧版默认流程
            celue_mod = _ensure_celue()
            celue2 = celue_mod.策略2(df_slice, HS300_信号, start_date=start_date, end_date=end_date)
            celue_sell = celue_mod.卖策略(df_slice, celue2, start_date=start_date, end_date=end_date)
            slice_index = df_slice.index
            celue2 = _align_bool(celue2, slice_index)
            celue_sell = _align_bool(celue_sell, slice_index)
            df.loc[start_date:end_date, 'celue_buy'] = celue2.values
            df.loc[start_date:end_date, 'celue_sell'] = celue_sell.values
            df.loc[start_date:end_date, 'celue_strategies'] = np.where(
                celue2.values, '策略2', ''
            )

        df.reset_index(drop=True, inplace=True)
        df.to_csv(ucfg.tdx['csv_lday'] + os.sep + stockcode + '.csv', index=False, encoding='gbk')
        df.to_pickle(pklfile)

    # 返回触发信号的行
    mask = df['celue_buy'].astype(bool) | df['celue_sell'].astype(bool)
    df_celue = df.loc[mask].copy()
    df_celue['date'] = pd.to_datetime(df_celue['date'], format='%Y-%m-%d')
    df_celue.set_index('date', drop=False, inplace=True)
    return df_celue


# ---------------------------------------------------------------------- #
# 多进程 Worker
# ---------------------------------------------------------------------- #
_HS300 = None
_FORCE_DEL = False
_ENGINE_CONFIG = None


def _pool_init(hs300_signal, force_del, engine_config):
    global _HS300, _FORCE_DEL, _ENGINE_CONFIG
    _HS300 = hs300_signal
    _FORCE_DEL = force_del
    _ENGINE_CONFIG = engine_config
    # 若使用多因子引擎，在子进程启动时主动校验策略函数可解析，
    # 让导入/配置错误在初始化阶段就明确抛出，而不是在每只股票上静默失败
    if engine_config is not None:
        try:
            StrategyEngine.from_dict(engine_config).validate()
        except Exception as e:
            print(f'[red]子进程策略配置校验失败: {e}[/red]')
            raise


def _pool_worker(args):
    file_list, tqdm_position = args
    if 'single' in sys.argv[1:]:
        tq = tqdm(file_list)
    else:
        tq = tqdm(file_list, leave=False, position=tqdm_position)
    df_celue = pd.DataFrame()
    for stockcode in tq:
        tq.set_description(stockcode)
        try:
            df_stock = compute_stock_signals(stockcode, _HS300, _FORCE_DEL, _ENGINE_CONFIG)
            df_celue = pd.concat([df_celue, df_stock])
        except Exception as e:
            print(f'[red]{stockcode} 处理失败: {e}[/red]')
    return df_celue


if __name__ == '__main__':
    args = parse_args()
    print(f'附带命令行参数 del 完全重新生成策略信号, 参数 single 单进程执行(默认多进程)')
    if args.force_del:
        print(f'检测到参数 del, 完全重新生成策略信号')
    if args.single:
        print(f'检测到参数 single, 单进程执行')
    if args.multi_factor:
        print(f'[green]启用多因子策略: {args.multi_factor}[/green]')
        engine = load_engine_from_json(args.multi_factor)
        # 在主进程先校验策略配置，使导入/配置错误在派生子进程前就明确暴露
        try:
            engine.validate()
        except Exception as e:
            print(f'[red]策略配置校验失败: {e}[/red]')
            sys.exit(2)
        engine_config = _engine_to_dict(engine.root)
    else:
        engine_config = None  # None 表示使用旧版默认策略（策略2 + 卖策略）

    db_path = get_db_path(args.db)
    print(f'SQLite 数据库路径: {db_path}')
    init_db(db_path)

    starttime = time.time()
    df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv',
                           index_col=None, encoding='gbk', dtype={'code': str})
    df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')
    df_hs300.set_index('date', drop=False, inplace=True)
    HS300_信号 = _ensure_celue().策略HS300(df_hs300)
    stocklist = [i[:-4] for i in os.listdir(ucfg.tdx['pickle'])]

    if args.single:
        _pool_init(HS300_信号, args.force_del, engine_config)
        df_celue = _pool_worker((stocklist, None))
    else:
        # 工作进程数
        if args.workers > 0:
            t_num = args.workers
        elif os.cpu_count() > 8:
            t_num = int(os.cpu_count() / 1.5)
        else:
            t_num = max(1, os.cpu_count() - 2)

        freeze_support()  # for Windows support
        tqdm.set_lock(RLock())
        p = Pool(processes=t_num, initializer=_pool_init,
                 initargs=(HS300_信号, args.force_del, engine_config))
        pool_result = []
        div = int(len(stocklist) / t_num)
        mod = len(stocklist) % t_num
        for i in range(0, t_num):
            if i + 1 != t_num:
                chunk = stocklist[i * div:(i + 1) * div]
            else:
                chunk = stocklist[i * div:(i + 1) * div + mod]
            pool_result.append(p.apply_async(_pool_worker, args=((chunk, i),)))
        p.close()
        p.join()

        df_list = []
        for i in pool_result:
            df_list.append(i.get())
        df_celue = pd.concat(df_list) if df_list else pd.DataFrame()

    # 自定义股票板块剔除
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
    df_celue = df_celue[~df_celue['code'].isin(kicklist)]

    # 保存到 SQLite
    print(f'保存策略信号到 SQLite: {db_path}')
    save_df_to_db(db_path, df_celue)

    # 同时保存 CSV 以兼容旧版回测流程
    if not args.no_csv:
        print(f'保存独立 "celue汇总.csv" 文件')
        keep_cols = ['code', 'date', 'close', 'celue_buy', 'celue_sell', 'celue_strategies', 'celue_score']
        existing_cols = [c for c in keep_cols if c in df_celue.columns]
        df_csv = (df_celue[existing_cols]
                  .sort_index()
                  .reset_index(drop=True))
        df_csv.to_csv(ucfg.tdx['csv_gbbq'] + os.sep + 'celue汇总.csv',
                      index=True, encoding='gbk')
    print(f'用时 {(time.time() - starttime):.2f} 秒, 全部处理完成，程序退出')
