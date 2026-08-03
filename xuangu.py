"""
选股多线程版本文件。导入数据——执行策略——显示结果
为保证和通达信选股一致，需使用前复权数据

多因子选股引擎支持（strategy_engine）：
  --multi-factor <json路径>   使用 JSON 配置的多因子组合策略（支持 AND/OR/NOT 嵌套、权重评分）
  --show-score                多因子模式下打印每只股票命中的子策略与综合评分
不指定 --multi-factor 时，行为与旧版完全一致（策略1(fast) + 策略2 两阶段筛选，AND 关系）。
旧版位置参数 single 仍然兼容。
"""
import os
import sys
import time
import argparse
import pandas as pd
from multiprocessing import Pool, RLock, freeze_support
from rich import print
from tqdm import tqdm
import CeLue  # 个人策略文件，不分享
import func
import user_config as ucfg
from strategy_engine import StrategyEngine, load_engine_from_json

# 配置部分

start_date = ''
end_date = ''

# 变量定义
tdxpath = ucfg.tdx['tdx_path']
csvdaypath = ucfg.tdx['pickle']
已选出股票列表 = []  # 策略选出的股票
要剔除的通达信概念 = ["ST板块", ]  # list类型。通达信软件中查看“概念板块”。
要剔除的通达信行业 = ["T1002", ]  # list类型。记事本打开 通达信目录\incon.dat，查看#TDXNHY标签的行业代码。

starttime_str = time.strftime("%H:%M:%S", time.localtime())
starttime = time.time()
starttime_tick = time.time()


def parse_args():
    """解析命令行启动参数，同时兼容旧版 single 位置参数。"""
    parser = argparse.ArgumentParser(
        description='股票选股程序（支持单策略与多因子组合策略）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('legacy_args', nargs='*',
                        help='兼容旧版位置参数：single(单进程)')
    parser.add_argument('--single', action='store_true',
                        help='单进程执行（默认多进程）')
    parser.add_argument('--multi-factor', dest='multi_factor', default=None,
                        help='多因子策略 JSON 配置文件路径。不指定则使用旧版默认单策略组合')
    parser.add_argument('--show-score', dest='show_score', action='store_true',
                        help='多因子模式下，输出每只股票命中的子策略及综合评分')
    args = parser.parse_args()
    # 兼容旧版位置参数
    if 'single' in args.legacy_args:
        args.single = True
    return args


def make_stocklist():
    # 要进行策略的股票列表筛选
    stocklist = [i[:-4] for i in os.listdir(ucfg.tdx['csv_lday'])]  # 去文件名里的.csv，生成纯股票代码list
    print(f'生成股票列表, 共 {len(stocklist)} 只股票')
    print(f'剔除通达信概念股票: {要剔除的通达信概念}')
    tmplist = []
    # 尝试读取通达信板块文件，如果不存在则跳过
    try:
        df = func.get_TDX_blockfilecontent("block_gn.dat")
        for i in 要剔除的通达信概念:
            tmplist = tmplist + df.loc[df['blockname'] == i]['code'].tolist()
        stocklist = list(filter(lambda i: i not in tmplist, stocklist))
    except FileNotFoundError:
        print("  通达信板块文件不存在，跳过概念剔除")
    print(f'剔除通达信行业股票: {要剔除的通达信行业}')
    tmplist = []
    try:
        df = pd.read_csv(ucfg.tdx['tdx_path'] + os.sep + 'T0002' + os.sep + 'hq_cache' + os.sep + "tdxhy.cfg",
                         sep='|', header=None, dtype='object')
        for i in 要剔除的通达信行业:
            tmplist = tmplist + df.loc[df[2] == i][1].tolist()
        stocklist = list(filter(lambda i: i not in tmplist, stocklist))
    except FileNotFoundError:
        print("  通达信行业文件不存在，跳过行业剔除")
    print("剔除科创板股票")
    tmplist = []
    for stockcode in stocklist:
        if stockcode[:2] != '68':
            tmplist.append(stockcode)
    stocklist = tmplist
    return stocklist


def load_dict_stock(stocklist):
    dicttemp = {}
    starttime_tick = time.time()
    tq = tqdm(stocklist)
    for stockcode in tq:
        tq.set_description(stockcode)
        pklfile = csvdaypath + os.sep + stockcode + '.pkl'
        # dict[stockcode] = pd.read_csv(csvfile, encoding='gbk', index_col=None, dtype={'code': str})
        dicttemp[stockcode] = pd.read_pickle(pklfile)
    print(f'载入完成 用时 {(time.time() - starttime_tick):.2f} 秒')
    return dicttemp


def _load_stock_df(stockcode, df_today=None, df_gbbq=None, update_realtime=True):
    """读取单只股票 pkl，按需合并今日实时行情并处理权息日复权，返回以 date 为索引的 DataFrame。"""
    pklfile = csvdaypath + os.sep + stockcode + '.pkl'
    df_stock = pd.read_pickle(pklfile)
    if update_realtime and df_today is not None:
        df_stock = func.update_stockquote(stockcode, df_stock, df_today)
        # 实时行情下若今天为权息日，需要重新前复权
        if df_gbbq is not None and '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00' \
                and 0 <= time.localtime(time.time()).tm_wday <= 4:
            now_date = pd.to_datetime(time.strftime("%Y-%m-%d", time.localtime()))
            if now_date in df_gbbq.loc[df_gbbq['code'] == stockcode]['权息日'].to_list():
                cw_dict = func.readall_local_cwfile()
                df_stock = func.make_fq(stockcode, df_stock, df_gbbq, cw_dict)
    df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')
    df_stock.set_index('date', drop=False, inplace=True)
    return df_stock


def run_celue1(stocklist, df_today, tqdm_position=None):
    """旧版策略1快速筛选（mode=fast）。保留原行为，返回通过筛选的股票列表。"""
    # 注意：旧版通过 stocklist.remove 原地修改列表存在索引跳变 bug，这里改为收集后返回
    if 'single' in sys.argv[1:]:
        tq = tqdm(stocklist[:])
    else:
        tq = tqdm(stocklist[:], leave=False, position=tqdm_position)
    选出 = []
    for stockcode in tq:
        tq.set_description(stockcode)
        pklfile = csvdaypath + os.sep + stockcode + '.pkl'
        df_stock = pd.read_pickle(pklfile)
        if df_today is not None:  # 更新当前最新行情，否则用昨天的数据
            df_stock = func.update_stockquote(stockcode, df_stock, df_today)
        df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')
        df_stock.set_index('date', drop=False, inplace=True)
        celue1 = CeLue.策略1(df_stock, start_date=start_date, end_date=end_date, mode='fast')
        if celue1:
            选出.append(stockcode)
    return 选出


def run_celue2(stocklist, HS300_信号, df_gbbq, df_today, tqdm_position=None):
    """旧版策略2详细筛选。保留原行为，返回通过筛选的股票列表。"""
    if 'single' in sys.argv[1:]:
        tq = tqdm(stocklist[:])
    else:
        tq = tqdm(stocklist[:], leave=False, position=tqdm_position)
    选出 = []
    for stockcode in tq:
        tq.set_description(stockcode)
        pklfile = csvdaypath + os.sep + stockcode + '.pkl'
        df_stock = pd.read_pickle(pklfile)
        df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')
        df_stock.set_index('date', drop=False, inplace=True)
        if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00' \
                and 0 <= time.localtime(time.time()).tm_wday <= 4:
            df_today_code = df_today.loc[df_today['code'] == stockcode]
            df_stock = func.update_stockquote(stockcode, df_stock, df_today_code)
            now_date = pd.to_datetime(time.strftime("%Y-%m-%d", time.localtime()))
            if now_date in df_gbbq.loc[df_gbbq['code'] == stockcode]['权息日'].to_list():
                cw_dict = func.readall_local_cwfile()
                df_stock = func.make_fq(stockcode, df_stock, df_gbbq, cw_dict)
        celue2 = CeLue.策略2(df_stock, HS300_信号, start_date=start_date, end_date=end_date).iat[-1]
        if celue2:
            选出.append(stockcode)
    return 选出


# ---------------------------------------------------------------------- #
# 多因子引擎的多进程 worker（必须定义在模块顶层，便于 pickle）
# ---------------------------------------------------------------------- #
_engine = None
_HS300 = None
_df_today = None
_df_gbbq = None


def _pool_init(engine_config, hs300_signal, df_today, df_gbbq):
    """子进程初始化：根据 dict 配置重建策略引擎，缓存共享行情数据。"""
    global _engine, _HS300, _df_today, _df_gbbq
    if engine_config is not None:
        _engine = StrategyEngine.from_dict(engine_config)
    else:
        _engine = StrategyEngine.default_single_strategy()
    _HS300 = hs300_signal
    _df_today = df_today
    _df_gbbq = df_gbbq


def _pool_worker(args):
    """多因子引擎 worker，args=(stocklist_chunk, tqdm_position)。"""
    stocklist, tqdm_position = args
    tq = tqdm(stocklist, leave=False, position=tqdm_position)
    results = []
    for stockcode in tq:
        tq.set_description(stockcode)
        try:
            df_stock = _load_stock_df(stockcode, df_today=_df_today, df_gbbq=_df_gbbq, update_realtime=True)
            r = _engine.evaluate_stock(df_stock, hs300_signal=_HS300,
                                       start_date=start_date, end_date=end_date)
            r['stock'] = stockcode
            results.append(r)
        except Exception as e:
            results.append({'stock': stockcode, 'matched': False, 'matched_strategies': [],
                            'score': 0.0, 'score_normalized': 0.0, 'error': str(e)})
    return results


def _engine_to_dict(node):
    """把策略树节点递归转为可 JSON 序列化的 dict（用于多进程传递配置）。"""
    from strategy_engine import Strategy, CompositeStrategy
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


def run_engine(stocklist, engine, HS300_信号, df_today, df_gbbq, single_process=False):
    """使用多因子引擎执行选股，返回每只股票的结果 dict 列表。"""
    engine_config = _engine_to_dict(engine.root) if (engine is not None and engine.root is not None) else None

    if single_process:
        _pool_init(engine_config, HS300_信号, df_today, df_gbbq)
        return _pool_worker((stocklist, None))

    if os.cpu_count() > 8:
        t_num = int(os.cpu_count() / 1.5)
    else:
        t_num = max(1, os.cpu_count() - 2)

    freeze_support()
    tqdm.set_lock(RLock())
    p = Pool(processes=t_num, initializer=_pool_init,
             initargs=(engine_config, HS300_信号, df_today, df_gbbq))
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

    all_results = []
    for r in pool_result:
        all_results.extend(r.get())
    return all_results


# 主程序开始
if __name__ == '__main__':
    args = parse_args()

    if args.single:
        print(f'检测到参数 single, 单进程执行')
    else:
        print(f'附带命令行参数 --single 单进程执行(默认多进程)')

    if args.multi_factor:
        print(f'[green]启用多因子组合策略引擎，配置文件: {args.multi_factor}[/green]')
        engine = load_engine_from_json(args.multi_factor)
        use_multi_factor = True
    else:
        print('未指定 --multi-factor，使用旧版默认策略（策略1(fast) AND 策略2）')
        engine = None
        use_multi_factor = False

    stocklist = make_stocklist()
    print(f'共 {len(stocklist)} 只候选股票')

    df_gbbq = pd.read_csv(ucfg.tdx['csv_gbbq'] + '/gbbq.csv', encoding='gbk', dtype={'code': str})

    # 策略部分
    # 先判断今天是否买入
    print('今日HS300行情判断')
    df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv', index_col=None, encoding='gbk', dtype={'code': str})
    df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')
    df_hs300.set_index('date', drop=False, inplace=True)
    if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00':
        df_today = func.get_tdx_lastestquote((1, '000300'))
        df_hs300 = func.update_stockquote('000300', df_hs300, df_today)
        del df_today
    HS300_信号 = CeLue.策略HS300(df_hs300)
    if HS300_信号.iat[-1]:
        print('[red]今日HS300满足买入条件，执行买入操作[/red]')
    else:
        print('[green]今日HS300不满足买入条件，仍然选股，但不执行买入操作[/green]')
        HS300_信号.loc[:] = True  # 强制全部设置为True出选股结果

    # 周一到周五，9点到16点之间，获取在线行情。其他时间不是交易日，默认为离线数据已更新到最新
    df_today_tmppath = ucfg.tdx['csv_gbbq'] + '/df_today.pkl'
    if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00' \
            and 0 <= time.localtime(time.time()).tm_wday <= 4:
        print(f'现在是交易时段，需要获取股票实时行情')
        if os.path.exists(df_today_tmppath):
            if round(time.time() - os.path.getmtime(df_today_tmppath)) < 600:
                print(f'检测到本地临时最新行情文件，读取并合并股票数据')
                df_today = pd.read_pickle(df_today_tmppath)
            else:
                df_today = func.get_tdx_lastestquote(stocklist)
                df_today.to_pickle(df_today_tmppath, compression=None)
        else:
            df_today = func.get_tdx_lastestquote(stocklist)
            df_today.to_pickle(df_today_tmppath, compression=None)
    else:
        try:
            os.remove(df_today_tmppath)
        except FileNotFoundError:
            pass
        df_today = None

    # ------------------------------------------------------------------ #
    # 多因子引擎模式：一次调用即可完成组合策略计算与评分
    # ------------------------------------------------------------------ #
    if use_multi_factor:
        print(f'开始执行多因子组合策略引擎')
        starttime_tick = time.time()
        results = run_engine(stocklist, engine, HS300_信号, df_today, df_gbbq,
                             single_process=args.single)
        # 只保留命中的股票
        matched_results = [r for r in results if r.get('matched')]
        # 按综合评分降序排列
        matched_results.sort(key=lambda x: x.get('score_normalized', 0), reverse=True)
        已选出股票列表 = [r['stock'] for r in matched_results]

        print(f'多因子策略执行完毕，已选出 {len(已选出股票列表):>d} 只股票 用时 {(time.time() - starttime_tick):>.2f} 秒')
        if args.show_score:
            print('[cyan]股票代码  综合评分  原始得分  命中子策略[/cyan]')
            for r in matched_results:
                print(f"{r['stock']}    {r['score_normalized']:>6.2f}    {r['score']:>6.2f}    {','.join(r['matched_strategies'])}")
        print(f'全部完成 共用时 {(time.time() - starttime):>.2f} 秒 已选出 {len(已选出股票列表)} 只股票:')
        print(已选出股票列表)
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # 旧版两阶段策略流程（完全保留原逻辑）
    # ------------------------------------------------------------------ #
    print(f'开始执行策略1(mode=fast)')
    starttime_tick = time.time()
    if args.single:
        stocklist = run_celue1(stocklist, df_today)
    else:
        if os.cpu_count() > 8:
            t_num = int(os.cpu_count() / 1.5)
        else:
            t_num = os.cpu_count() - 2
        freeze_support()
        tqdm.set_lock(RLock())
        p = Pool(processes=t_num, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        pool_result = []
        for i in range(0, t_num):
            div = int(len(stocklist) / t_num)
            mod = len(stocklist) % t_num
            if i + 1 != t_num:
                pool_result.append(p.apply_async(run_celue1, args=(stocklist[i * div:(i + 1) * div], df_today, i,)))
            else:
                pool_result.append(p.apply_async(run_celue1, args=(stocklist[i * div:(i + 1) * div + mod], df_today, i,)))
        p.close()
        p.join()
        stocklist = []
        for i in pool_result:
            stocklist = stocklist + i.get()

    print(f'策略1执行完毕，已选出 {len(stocklist):>d} 只股票 用时 {(time.time() - starttime_tick):>.2f} 秒')

    print(f'开始执行策略2')
    if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00' and df_today is None:
        df_today = func.get_tdx_lastestquote(stocklist)

    starttime_tick = time.time()
    if args.single:
        stocklist = run_celue2(stocklist, HS300_信号, df_gbbq, df_today)
    else:
        t_num = os.cpu_count() - 2
        freeze_support()
        tqdm.set_lock(RLock())
        p = Pool(processes=t_num, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        pool_result = []
        for i in range(0, t_num):
            div = int(len(stocklist) / t_num)
            mod = len(stocklist) % t_num
            if i + 1 != t_num:
                pool_result.append(p.apply_async(run_celue2, args=(stocklist[i * div:(i + 1) * div], HS300_信号, df_gbbq, df_today, i,)))
            else:
                pool_result.append(p.apply_async(run_celue2, args=(stocklist[i * div:(i + 1) * div + mod], HS300_信号, df_gbbq, df_today, i,)))
        p.close()
        p.join()
        stocklist = []
        for i in pool_result:
            stocklist = stocklist + i.get()

    print(f'策略2执行完毕，已选出 {len(stocklist):>d} 只股票 用时 {(time.time() - starttime_tick):>.2f} 秒')

    # 结果
    已选出股票列表 = stocklist
    print(f'全部完成 共用时 {(time.time() - starttime):>.2f} 秒 已选出 {len(stocklist)} 只股票:')
    print(stocklist)
