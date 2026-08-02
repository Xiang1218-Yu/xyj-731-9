"""
实时行情监控与预警模块

功能：
1. 每30秒（可用 --interval 或 user_config.monitor['interval'] 配置）从通达信获取一次监控列表的实时行情
2. 对监控列表中的股票持续运行选股策略（单策略模式=CeLue.策略2/卖策略，--multi 参数=多因子引擎）。
   每轮检测最近 lookback 个周期（默认3，可用 --lookback 配置），持续运行中触发过的信号不会遗漏
3. 当某只股票触发买入/卖出信号时，记录信号时间、代码、信号日期、价格、触发策略，
   写入预警日志文件(user_config.monitor['alert_log'])和信号CSV(user_config.monitor['alert_csv'])
4. 支持 start/stop/pause/resume 控制。监控在后台线程运行，start() 立即返回，不阻塞主进程

用法：
    python3 monitor.py                          # 交互控制台模式（start/stop/pause/resume/status/quit）
    python3 monitor.py --watch 000001,600030    # 指定监控股票列表
    python3 monitor.py --interval 30 --multi    # 指定行情间隔秒数，并使用多因子引擎判定买入信号
    python3 monitor.py --lookback 5 --start-date 2020-01-01  # 指定信号检测周期数与策略计算区间
    python3 monitor.py --once                   # 只执行一次检查（调试用）

在其他程序中嵌入（不阻塞主进程）：
    from monitor import StockMonitor
    m = StockMonitor(['000001', '600030'], interval=30)
    m.start()   # 后台线程运行，立即返回
    ...         # 主进程继续做其他事
    m.pause(); m.resume(); m.stop()
"""
import argparse
import logging
import os
import sys
import threading
import time

import pandas as pd
from rich import print

import CeLue  # 个人策略文件，不分享
import celue_engine  # 多因子选股策略引擎
import func
import user_config as ucfg


class StockMonitor:
    """实时行情监控器。后台线程持续获取行情并运行策略，触发信号时生成预警"""

    def __init__(self, watchlist, interval=None, use_multi_factor=None,
                 lookback=None, start_date=None, end_date=None):
        """
        :param watchlist: 监控股票代码列表，如 ['000001', '600030']
        :param interval: 行情获取间隔秒数，默认读取 user_config.monitor['interval']（30秒）
        :param use_multi_factor: True=买入信号用多因子引擎判定，False=原策略2单策略
        :param lookback: 每轮检测最近N个周期的信号，默认读取 user_config.monitor['lookback']
        :param start_date: 策略计算起始日期，空=从数据最早日期开始
        :param end_date: 策略计算截止日期，空=到最新数据
        """
        self.watchlist = list(watchlist)
        self.interval = interval if interval is not None else ucfg.monitor.get('interval', 30)
        self.use_multi_factor = (use_multi_factor if use_multi_factor is not None
                                 else ucfg.monitor.get('use_multi_factor', False))
        self.lookback = lookback if lookback is not None else ucfg.monitor.get('lookback', 3)
        self.start_date = start_date if start_date is not None else ucfg.monitor.get('start_date', '')
        self.end_date = end_date if end_date is not None else ucfg.monitor.get('end_date', '')
        self.alert_csv = ucfg.monitor.get('alert_csv', 'monitor_signals.csv')  # 预警信号记录文件
        self.alert_log = ucfg.monitor.get('alert_log', 'monitor_alert.log')  # 预警日志文件

        self._thread = None  # 监控后台线程
        self._stop_event = threading.Event()  # 停止标志
        self._pause_event = threading.Event()  # 暂停标志。set=暂停
        self._state = 'stopped'  # 当前状态 running/paused/stopped
        self._alerted = set()  # 已预警去重集合 (code, signal, date)
        self._hs300_cache = {'date': None, '信号': None}  # HS300信号按天缓存
        self._engine = celue_engine.MultiFactorEngine() if self.use_multi_factor else None

        # 预警日志器：同时输出到控制台和日志文件
        self.logger = logging.getLogger('StockMonitor')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            fh = logging.FileHandler(self.alert_log, encoding='utf-8')
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)

    # ==================== 控制接口 ====================

    def start(self):
        """启动监控。后台线程运行，立即返回，不阻塞主进程"""
        if self._thread is not None and self._thread.is_alive():
            self.logger.info('监控已在运行中')
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._loop, name='StockMonitor', daemon=True)
        self._thread.start()
        self._state = 'running'
        self.logger.info(f'监控已启动 股票数={len(self.watchlist)} 间隔={self.interval}秒 '
                         f'多因子={self.use_multi_factor}')

    def stop(self):
        """停止监控。等待后台线程退出"""
        self._stop_event.set()
        self._pause_event.clear()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._state = 'stopped'
        self.logger.info('监控已停止')

    def pause(self):
        """暂停监控（保留线程，停止行情获取与策略计算）"""
        if self._state == 'running':
            self._pause_event.set()
            self._state = 'paused'
            self.logger.info('监控已暂停')

    def resume(self):
        """从暂停状态恢复监控"""
        if self._state == 'paused':
            self._pause_event.clear()
            self._state = 'running'
            self.logger.info('监控已恢复')

    @property
    def state(self):
        return self._state

    # ==================== 监控主循环 ====================

    def _loop(self):
        """后台线程主循环：每interval秒执行一次检查，停止/暂停可即时响应"""
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(1)  # 暂停中，低功耗等待
                continue
            try:
                self.check_once()
            except Exception as e:
                # 单次检查异常不中断监控，记录日志后继续
                self.logger.exception(f'本轮检查异常: {e}')
            # 可中断的等待，保证stop()能及时生效
            self._stop_event.wait(self.interval)
        self.logger.info('监控线程退出')

    def _get_hs300(self):
        """构建HS300大盘信号，按天缓存避免重复计算"""
        today = time.strftime('%Y-%m-%d', time.localtime())
        if self._hs300_cache['date'] == today:
            return self._hs300_cache['信号']
        df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv', index_col=None,
                               encoding='gbk', dtype={'code': str})
        df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')  # 转为时间格式
        df_hs300.set_index('date', drop=False, inplace=True)  # 时间为索引
        hs300 = CeLue.策略HS300(df_hs300)
        self._hs300_cache = {'date': today, '信号': hs300}
        return hs300

    def check_once(self):
        """执行一轮检查：获取实时行情 -> 逐股持续运行策略 -> 检测最近lookback个周期触发的信号并预警"""
        tick = time.time()
        # 从通达信获取监控列表实时行情。网络异常时返回空DF，本轮用昨日离线数据评估
        df_today = func.get_tdx_lastestquote(self.watchlist)
        if df_today is None or len(df_today) == 0:
            self.logger.warning('未获取到实时行情（非交易时段或网络异常），本轮使用本地最新数据评估')
            df_today = None
        hs300 = self._get_hs300()

        for code in self.watchlist:
            pklfile = ucfg.tdx['pickle'] + os.sep + code + '.pkl'
            if not os.path.exists(pklfile):
                self.logger.warning(f'{code} 本地日线数据不存在，跳过')
                continue
            df_stock = pd.read_pickle(pklfile)
            df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')  # 转为时间格式
            df_stock.set_index('date', drop=False, inplace=True)  # 时间为索引
            if df_today is not None:
                # 用盘中实时行情更新该股当日数据（只在内存操作，不保存文件）
                df_stock = func.update_stockquote(code, df_stock,
                                                  df_today.loc[df_today['code'] == code])
            context = {'HS300_信号': hs300,
                       'start_date': self.start_date, 'end_date': self.end_date}
            # 检测区间：最近lookback个周期。持续运行中任何一天触发过的信号都不会遗漏
            bars = df_stock.index[-self.lookback:]

            # ---- 买入信号检测（逐周期判定，命中未预警过的周期即预警） ----
            if self.use_multi_factor:
                # 多因子模式：逐周期进行逻辑组合+评分门槛判定，触发策略=该周期匹配的子策略列表
                # 传入context本体（非副本），引擎计算的策略2序列缓存可供卖出检测复用
                df_eval = self._engine.evaluate_series(code, df_stock, context,
                                                       lookback=self.lookback)
                for bar, row in df_eval.iterrows():
                    if row['passed']:
                        trigger = row['matched_strategies'].split('|') if row['matched_strategies'] else []
                        self._alert(code, '买入', self._price_at(df_stock, bar), trigger, bar)
                # 策略2序列可能在引擎上下文中已缓存（卖策略因子启用时）
                celue2_series = context.get('_cache', {}).get('策略2')
                if celue2_series is None:
                    celue2_series = self._single_celue2(df_stock, context)
            else:
                # 单策略模式：沿用原策略2判定
                celue2_series = self._single_celue2(df_stock, context)
                for bar in bars:
                    if bool(celue2_series.get(bar, False)):
                        self._alert(code, '买入', self._price_at(df_stock, bar), ['策略2'], bar)

            # ---- 卖出信号检测（卖策略入参与CeLue.py签名一致，显式传递start_date/end_date） ----
            sell_series = CeLue.卖策略(df_stock, celue2_series,
                                       start_date=self.start_date, end_date=self.end_date)
            sell_series = celue_engine.MultiFactorEngine._normalize(sell_series, df_stock.index)
            for bar in bars:
                if bool(sell_series.get(bar, False)):
                    self._alert(code, '卖出', self._price_at(df_stock, bar), ['卖策略'], bar)

        self.logger.info(f'本轮检查完成 用时 {(time.time() - tick):.2f} 秒')

    def _single_celue2(self, df_stock, context):
        """计算策略2买入信号完整序列并归一化到df索引，供买入/卖出检测共用"""
        celue2_series = celue_engine.run_single_strategy('策略2', df_stock, context)
        return celue_engine.MultiFactorEngine._normalize(celue2_series, df_stock.index)

    @staticmethod
    def _price_at(df_stock, bar):
        """取指定周期收盘价（前复权，盘中最后一根为实时价）"""
        return round(float(df_stock['close'].at[bar]), 2)

    def _alert(self, code, signal, price, trigger_strategies, bar):
        """生成预警：同一股票同一信号在同一信号周期只预警一次。写预警日志+信号CSV"""
        bar_str = pd.to_datetime(bar).strftime('%Y-%m-%d')  # 信号所在交易日
        key = (code, signal, bar_str)
        if key in self._alerted:
            return
        self._alerted.add(key)
        trigger = '|'.join(trigger_strategies)
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        # 预警日志：预警时间、股票代码、信号类型、信号日期、价格、触发策略
        self.logger.warning(f'[{signal}预警] {code} 信号日期 {bar_str} 价格 {price} 触发策略: {trigger}')
        # 信号CSV：供其他程序（如自动化交易）读取
        row = pd.DataFrame([{'signal_time': now, 'code': code, 'signal': signal, 'signal_date': bar_str,
                             'price': price, 'trigger_strategy': trigger}])
        row.to_csv(self.alert_csv, mode='a', index=False,
                   header=not os.path.exists(self.alert_csv), encoding='gbk')


def load_default_watchlist():
    """
    默认监控列表。优先级：user_config.monitor['watchlist'] > multi_result.csv（多因子选股结果）
    > celue汇总.csv 最近一个交易日策略2买入信号股票
    """
    if ucfg.monitor.get('watchlist'):
        return list(ucfg.monitor['watchlist'])
    if os.path.exists('multi_result.csv'):
        df = pd.read_csv('multi_result.csv', encoding='gbk', dtype={'code': str})
        codes = df['code'].tolist()
        print(f'使用 multi_result.csv 中的 {len(codes)} 只股票作为监控列表')
        return codes
    celue_csv = ucfg.tdx['csv_gbbq'] + os.sep + 'celue汇总.csv'
    if os.path.exists(celue_csv):
        df = pd.read_csv(celue_csv, index_col=0, encoding='gbk', dtype={'code': str})
        last_date = df['date'].max()
        codes = df.loc[(df['date'] == last_date) & (df['celue_buy'] == True)]['code'].tolist()
        print(f'使用 celue汇总.csv {last_date} 买入信号的 {len(codes)} 只股票作为监控列表')
        return codes
    return []


def main():
    parser = argparse.ArgumentParser(description='实时行情监控与预警模块')
    parser.add_argument('--watch', type=str, default='',
                        help='监控股票列表，逗号分隔，如 000001,600030。默认读取user_config.monitor配置')
    parser.add_argument('--interval', type=int, default=None,
                        help='行情获取间隔秒数，默认30秒')
    parser.add_argument('--multi', action='store_true',
                        help='使用多因子引擎判定买入信号（配置见user_config.multi_factor）')
    parser.add_argument('--lookback', type=int, default=None,
                        help='每轮检测最近N个周期的信号，默认3')
    parser.add_argument('--start-date', type=str, default=None,
                        help='策略计算起始日期，如 2020-10-10，默认从数据最早日期开始')
    parser.add_argument('--end-date', type=str, default=None,
                        help='策略计算截止日期，如 2020-10-10，默认到最新数据')
    parser.add_argument('--once', action='store_true',
                        help='只执行一次检查后退出（调试用）')
    args = parser.parse_args()

    watchlist = [c.strip() for c in args.watch.split(',') if c.strip()] or load_default_watchlist()
    if not watchlist:
        print('监控列表为空，请用 --watch 指定股票，或在 user_config.monitor 中配置 watchlist')
        sys.exit(1)

    monitor = StockMonitor(watchlist,
                           interval=args.interval,
                           use_multi_factor=True if args.multi else None,
                           lookback=args.lookback,
                           start_date=args.start_date,
                           end_date=args.end_date)

    if args.once:  # 单次检查模式（调试）
        monitor.check_once()
        return

    # 交互控制台模式。监控在后台线程运行，主线程处理控制命令，互不阻塞
    monitor.start()
    print('监控运行中。可用命令: start / stop / pause / resume / status / quit')
    try:
        while True:
            cmd = input('monitor> ').strip().lower()
            if cmd == 'start':
                monitor.start()
            elif cmd == 'stop':
                monitor.stop()
            elif cmd == 'pause':
                monitor.pause()
            elif cmd == 'resume':
                monitor.resume()
            elif cmd == 'status':
                print(f'状态: {monitor.state} 监控股票数: {len(monitor.watchlist)} '
                      f'间隔: {monitor.interval}秒')
            elif cmd in ('quit', 'exit', 'q'):
                break
            elif cmd:
                print('未知命令。可用命令: start / stop / pause / resume / status / quit')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        monitor.stop()
    print('监控程序退出')


if __name__ == '__main__':
    main()
