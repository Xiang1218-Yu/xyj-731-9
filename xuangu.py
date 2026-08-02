"""
选股多线程版本文件。导入数据——执行策略——显示结果
为保证和通达信选股一致，需使用前复权数据

命令行参数：
  single              单进程执行（默认多进程）
  --multi <file>      使用多因子策略引擎，从 JSON 配置文件加载策略组合
                      配置格式见 strategy_engine.py 的 build_engine_from_config 说明
  --score             多因子模式下，按综合评分排序输出（默认按股票代码排序）
"""
import os
import sys
import json
import time
import pandas as pd
from multiprocessing import Pool, RLock, freeze_support
from rich import print
from tqdm import tqdm
import CeLue  # 个人策略文件，不分享
import func
import user_config as ucfg
import strategy_engine

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


def run_celue1(stocklist, df_today, tqdm_position=None):
    if 'single' in sys.argv[1:]:
        tq = tqdm(stocklist[:])
    else:
        tq = tqdm(stocklist[:], leave=False, position=tqdm_position)
    for stockcode in tq:
        tq.set_description(stockcode)
        pklfile = csvdaypath + os.sep + stockcode + '.pkl'
        df_stock = pd.read_pickle(pklfile)
        if df_today is not None:  # 更新当前最新行情，否则用昨天的数据
            df_stock = func.update_stockquote(stockcode, df_stock, df_today)
        df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')  # 转为时间格式
        df_stock.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并
        celue1 = CeLue.策略1(df_stock, start_date=start_date, end_date=end_date, mode='fast')
        if not celue1:
            stocklist.remove(stockcode)
    return stocklist


def run_celue2(stocklist, HS300_信号, df_gbbq, df_today, tqdm_position=None):
    if 'single' in sys.argv[1:]:
        tq = tqdm(stocklist[:])
    else:
        tq = tqdm(stocklist[:], leave=False, position=tqdm_position)
    for stockcode in tq:
        tq.set_description(stockcode)
        pklfile = csvdaypath + os.sep + stockcode + '.pkl'
        df_stock = pd.read_pickle(pklfile)
        df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')  # 转为时间格式
        df_stock.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并
        if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00' \
                and 0 <= time.localtime(time.time()).tm_wday <= 4:
            df_today_code = df_today.loc[df_today['code'] == stockcode]
            df_stock = func.update_stockquote(stockcode, df_stock, df_today_code)
            # 判断今天是否在该股的权息日内。如果是，需要重新前复权
            now_date = pd.to_datetime(time.strftime("%Y-%m-%d", time.localtime()))
            if now_date in df_gbbq.loc[df_gbbq['code'] == stockcode]['权息日'].to_list():
                cw_dict = func.readall_local_cwfile()
                df_stock = func.make_fq(stockcode, df_stock, df_gbbq, cw_dict)
        celue2 = CeLue.策略2(df_stock, HS300_信号, start_date=start_date, end_date=end_date).iat[-1]
        if not celue2:
            stocklist.remove(stockcode)
    return stocklist


# 多进程 worker 的全局引擎变量，通过 initializer 在每个子进程启动时构建一次，避免重复构建
_worker_engine = None
_worker_engine_config = None


def _init_worker(engine_config, lock):
    """进程池初始化函数：在每个子进程启动时构建一次策略引擎并复用"""
    global _worker_engine, _worker_engine_config
    tqdm.set_lock(lock)
    _worker_engine_config = engine_config
    if engine_config is not None:
        _worker_engine = strategy_engine.build_engine_from_config(engine_config)


def run_multi_factor_worker(stocklist, HS300_信号, df_gbbq, df_today,
                            start_date='', end_date='', tqdm_position=None):
    """
    多因子策略引擎工作函数，可在多进程中调用。
    引擎实例由 _init_worker 在进程启动时构建一次，通过全局变量复用，避免每只股票重复构建。
    返回 {stockcode: StrategyResult} 字典。
    """
    global _worker_engine
    if _worker_engine is None:
        # 单进程模式下 initializer 未执行，在此兜底构建
        _worker_engine = strategy_engine.build_engine_from_config(_worker_engine_config)

    results = {}
    tq = tqdm(stocklist[:], leave=False, position=tqdm_position)
    for stockcode in tq:
        tq.set_description(stockcode)
        pklfile = csvdaypath + os.sep + stockcode + '.pkl'
        try:
            df_stock = pd.read_pickle(pklfile)
        except FileNotFoundError:
            continue
        if df_today is not None:
            df_stock = func.update_stockquote(stockcode, df_stock, df_today)
        df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')
        df_stock.set_index('date', drop=False, inplace=True)
        context = {
            'start_date': start_date,
            'end_date': end_date,
        }
        if HS300_信号 is not None:
            context['HS300_信号'] = HS300_信号
        if df_gbbq is not None:
            context['df_gbbq'] = df_gbbq
        result = _worker_engine.evaluate(df_stock, **context)
        if result.matched:
            results[stockcode] = result
    return results


# 主程序开始
if __name__ == '__main__':
    # 解析命令行参数
    multi_config_path = None
    sort_by_score = False
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == '--multi' and i + 1 < len(argv):
            multi_config_path = argv[i + 1]
            i += 2
            continue
        elif argv[i] == '--score':
            sort_by_score = True
        i += 1

    use_multi_factor = multi_config_path is not None

    if use_multi_factor:
        print(f'[bold blue]多因子策略引擎模式[/bold blue]，配置文件: {multi_config_path}')
        if not os.path.exists(multi_config_path):
            print(f'[red]错误：配置文件 {multi_config_path} 不存在[/red]')
            sys.exit(1)
        with open(multi_config_path, 'r', encoding='utf-8') as f:
            engine_config = json.load(f)
    elif 'single' in sys.argv[1:]:
        print(f'检测到参数 single, 单进程执行')
    else:
        print(f'附带命令行参数 single 单进程执行(默认多进程)')

    stocklist = make_stocklist()
    print(f'共 {len(stocklist)} 只候选股票')

    # 由于多进程时df_dict字典占用超多内存资源，导致多进程效率还不如单进程。因此多进程模式改用函数内部读单独股票pkl文件的办法
    # print("开始载入日线文件到内存")
    # df_dict = load_dict_stock(stocklist)

    df_gbbq = pd.read_csv(ucfg.tdx['csv_gbbq'] + '/gbbq.csv', encoding='gbk', dtype={'code': str})

    # 策略部分
    # 先判断今天是否买入
    print('今日HS300行情判断')
    df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv', index_col=None, encoding='gbk', dtype={'code': str})
    df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')  # 转为时间格式
    df_hs300.set_index('date', drop=False, inplace=True)  # 时间为索引。方便与另外复权的DF表对齐合并
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
        # 获取当前最新行情，临时保存到本地，防止多次调用被服务器封IP。
        print(f'现在是交易时段，需要获取股票实时行情')
        if os.path.exists(df_today_tmppath):
            if round(time.time() - os.path.getmtime(df_today_tmppath)) < 600:  # 据创建时间小于10分钟读取本地文件
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

    # ====== 策略执行分支 ======
    if use_multi_factor:
        # 多因子策略引擎模式
        print(f'开始执行多因子策略引擎')
        starttime_tick = time.time()
        multi_results = {}

        if 'single' in sys.argv[1:]:
            # 单进程模式：设置全局配置后直接调用，worker 内部兜底构建一次引擎
            _worker_engine_config = engine_config
            multi_results = run_multi_factor_worker(
                stocklist, HS300_信号, df_gbbq, df_today,
                start_date=start_date, end_date=end_date,
            )
        else:
            # 多进程并行执行：通过 initializer 在每个子进程启动时构建一次引擎并复用
            if os.cpu_count() > 8:
                t_num = int(os.cpu_count() / 1.5)
            else:
                t_num = os.cpu_count() - 2
            freeze_support()
            tqdm.set_lock(RLock())
            # initializer 在每个子进程启动时执行一次，构建引擎实例并设置 tqdm 锁
            p = Pool(processes=t_num, initializer=_init_worker,
                    initargs=(engine_config, tqdm.get_lock()))
            pool_result = []
            for i in range(0, t_num):
                div = int(len(stocklist) / t_num)
                mod = len(stocklist) % t_num
                if i + 1 != t_num:
                    pool_result.append(p.apply_async(
                        run_multi_factor_worker,
                        args=(stocklist[i * div:(i + 1) * div],
                              HS300_信号, df_gbbq, df_today, start_date, end_date, i)))
                else:
                    pool_result.append(p.apply_async(
                        run_multi_factor_worker,
                        args=(stocklist[i * div:(i + 1) * div + mod],
                              HS300_信号, df_gbbq, df_today, start_date, end_date, i)))
            p.close()
            p.join()
            for pr in pool_result:
                multi_results.update(pr.get())

        print(f'多因子策略执行完毕，已选出 {len(multi_results)} 只股票 '
              f'用时 {(time.time() - starttime_tick):.2f} 秒')

        # 输出结果：股票代码、匹配子策略列表、综合评分
        result_rows = []
        for code, res in multi_results.items():
            result_rows.append({
                'code': code,
                'matched_strategies': ','.join(res.matched_strategies),
                'score': round(res.score, 3),
            })
        df_result = pd.DataFrame(result_rows)
        if len(df_result) > 0:
            if sort_by_score:
                df_result = df_result.sort_values(by='score', ascending=False).reset_index(drop=True)
            else:
                df_result = df_result.sort_values(by='code').reset_index(drop=True)
            print(f'\n全部完成 共用时 {(time.time() - starttime):.2f} 秒 已选出 {len(df_result)} 只股票:')
            print(df_result.to_string(index=False))
            stocklist = df_result['code'].tolist()
        else:
            print(f'\n全部完成 共用时 {(time.time() - starttime):.2f} 秒 没有选出任何股票')
            stocklist = []
    else:
        # ====== 原有单策略模式（保持向后兼容）======
        print(f'开始执行策略1(mode=fast)')
        starttime_tick = time.time()
        if 'single' in sys.argv[1:]:
            stocklist = run_celue1(stocklist, df_today)
        else:
            # 进程数 读取CPU逻辑处理器个数
            if os.cpu_count() > 8:
                t_num = int(os.cpu_count() / 1.5)
            else:
                t_num = os.cpu_count() - 2
            freeze_support()  # for Windows support
            tqdm.set_lock(RLock())  # for managing output contention
            p = Pool(processes=t_num, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
            pool_result = []  # 存放pool池的返回对象列表
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
        if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00' and 'df_today' not in dir():
            df_today = func.get_tdx_lastestquote(stocklist)

        starttime_tick = time.time()
        if 'single' in sys.argv[1:]:
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

        print(f'全部完成 共用时 {(time.time() - starttime):>.2f} 秒 已选出 {len(stocklist)} 只股票:')
        print(stocklist)
