#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
实时行情监控与预警模块
======================

对监控列表中的股票持续运行选股策略，实现盘中实时预警：

* 每隔 N 秒（默认 30，可配置）从通达信 pytdx 获取一次实时行情；
* 将实时行情合并进各股的历史日线数据（内存操作，不落盘），复用 func.update_stockquote；
* 对监控列表中的每只股票运行策略：
    - 买入信号：使用多因子组合引擎 CeLue_engine（配置见 user_config.strategy_combo）；
    - 卖出信号：使用 CeLue.卖策略；
* 触发买入/卖出信号时，记录信号时间、价格、触发策略，并写入预警日志。

控制能力：
* 监控在独立后台线程运行，不阻塞主进程；
* 支持 启动(start) / 停止(stop) / 暂停(pause) / 恢复(resume) 控制；
* 命令行运行时提供交互式控制台，也可作为库被其他程序 import 后编程控制。

启动参数（python monitor.py [参数...]）：
    once                 只跑一轮就退出（便于测试，不进入循环）
    interval=<秒>         覆盖轮询间隔，如 interval=10
    code=<代码,代码...>   指定监控股票列表，如 code=000001,600030
    nointeractive        非交互模式，后台线程运行直到 Ctrl+C

作者：功能迭代新增，命名与项目现有风格保持一致。
"""
import os
import sys
import time
import threading
import logging

import pandas as pd

import CeLue  # 个人策略文件，不分享。卖出信号复用 CeLue.卖策略
import CeLue_engine  # 多因子组合引擎，用于买入信号
import func
import user_config as ucfg


def _build_logger(log_file):
    """构建预警日志器，同时输出到文件和控制台"""
    logger = logging.getLogger('monitor')
    logger.setLevel(logging.INFO)
    if logger.handlers:  # 防止重复添加 handler
        return logger
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


class StockMonitor:
    """
    实时监控器。内部用一个后台线程循环拉取行情并跑策略，主线程不被阻塞。

    线程安全的状态控制：
        start()   启动后台线程
        pause()   暂停（线程仍在，只是跳过每轮工作）
        resume()  恢复
        stop()    停止并回收后台线程
    """

    def __init__(self, watchlist=None, interval=None, log_file=None):
        # 监控列表：为空则取 csv_lday 目录下全部股票
        if watchlist:
            self.watchlist = list(watchlist)
        elif ucfg.monitor.get('watchlist'):
            self.watchlist = list(ucfg.monitor['watchlist'])
        else:
            self.watchlist = [i[:-4] for i in os.listdir(ucfg.tdx['csv_lday'])]

        self.interval = interval or ucfg.monitor.get('interval', 30)
        self.log_file = log_file or ucfg.monitor.get('log_file', 'monitor_alert.log')
        self.logger = _build_logger(self.log_file)

        # 线程与状态控制
        self._thread = None
        self._stop_event = threading.Event()   # 置位表示需要停止
        self._pause_event = threading.Event()  # 置位表示暂停
        self._running = False

        # 组合买入引擎
        self.engine = CeLue_engine.StrategyEngine(ucfg.strategy_combo)

        # 预加载 HS300 信号与 gbbq，供策略使用
        self.HS300_信号 = self._load_hs300_signal()
        try:
            self.df_gbbq = pd.read_csv(ucfg.tdx['csv_gbbq'] + '/gbbq.csv',
                                       encoding='gbk', dtype={'code': str})
        except FileNotFoundError:
            self.df_gbbq = pd.DataFrame(columns=['code', '权息日'])

        # 记录每只股票上一次的信号，避免同一信号在多轮内重复预警刷屏
        self._last_signal = {}

    def _load_hs300_signal(self):
        """载入 HS300 买入许可信号，逻辑对齐 xuangu.py"""
        try:
            df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv',
                                   index_col=None, encoding='gbk', dtype={'code': str})
            df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')
            df_hs300.set_index('date', drop=False, inplace=True)
            return CeLue.策略HS300(df_hs300)
        except Exception as e:
            self.logger.warning(f'载入HS300信号失败，默认放行: {e}')
            return None

    # ---------------- 单只股票信号检测 ----------------
    def _check_stock(self, stockcode, df_today):
        """
        对单只股票合并实时行情并运行买卖策略。
        触发买入/卖出信号时写预警日志（去重：只在信号状态变化时报警）。
        """
        pklfile = ucfg.tdx['pickle'] + os.sep + stockcode + '.pkl'
        if not os.path.exists(pklfile):
            return
        df_stock = pd.read_pickle(pklfile)
        # 合并实时行情（内存操作，不落盘）
        if df_today is not None:
            df_today_code = df_today.loc[df_today['code'] == stockcode] \
                if 'code' in df_today.columns else df_today
            if len(df_today_code):
                df_stock = func.update_stockquote(stockcode, df_stock, df_today_code)
        df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')
        df_stock.set_index('date', drop=False, inplace=True)

        latest_price = df_stock['close'].iat[-1]
        now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        # 买入信号：多因子组合引擎
        try:
            context = {'HS300_信号': self.HS300_信号}
            ret = self.engine.run(df_stock, context)
        except Exception:
            ret = {'matched': False, 'matched_factors': [], 'score': 0.0}

        # 卖出信号：复用 CeLue.卖策略（需要策略2序列作为买点输入）
        sell_signal = False
        try:
            celue2 = CeLue.策略2(df_stock, self.HS300_信号)
            sell_series = CeLue.卖策略(df_stock, celue2)
            if len(sell_series):
                sell_signal = bool(sell_series.iloc[-1])
        except Exception:
            sell_signal = False

        prev = self._last_signal.get(stockcode)
        # 买入预警（仅在状态由非买入变为买入时报警）
        if ret['matched'] and prev != 'BUY':
            self.logger.info(
                f'[买入信号] {stockcode} 时间={now_str} 价格={latest_price:.2f} '
                f'触发策略={ret["matched_factors"]} 综合评分={ret["score"]}')
            self._last_signal[stockcode] = 'BUY'
        # 卖出预警
        elif sell_signal and prev != 'SELL':
            self.logger.info(
                f'[卖出信号] {stockcode} 时间={now_str} 价格={latest_price:.2f} 触发策略=卖策略')
            self._last_signal[stockcode] = 'SELL'
        elif not ret['matched'] and not sell_signal:
            self._last_signal[stockcode] = None

    # ---------------- 一轮扫描 ----------------
    def _run_once(self):
        """获取一次实时行情，并对监控列表全部股票检测信号"""
        # 交易时段才获取实时行情，非交易时段用离线数据
        if '09:00:00' < time.strftime('%H:%M:%S', time.localtime()) < '16:00:00' \
                and 0 <= time.localtime(time.time()).tm_wday <= 4:
            try:
                df_today = func.get_tdx_lastestquote(self.watchlist)
            except Exception as e:
                self.logger.warning(f'获取实时行情失败: {e}')
                df_today = None
        else:
            df_today = None

        for stockcode in self.watchlist:
            if self._stop_event.is_set():
                break
            try:
                self._check_stock(stockcode, df_today)
            except Exception as e:
                self.logger.warning(f'{stockcode} 检测异常: {e}')

    # ---------------- 后台循环 ----------------
    def _loop(self):
        self.logger.info(f'监控启动，监控 {len(self.watchlist)} 只股票，间隔 {self.interval} 秒')
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(1)  # 暂停态：轻量空转
                continue
            start_tick = time.time()
            self._run_once()
            # 扣除本轮耗时后按 interval 定时，用小步 sleep 以便及时响应停止
            elapsed = time.time() - start_tick
            wait = max(0, self.interval - elapsed)
            waited = 0.0
            while waited < wait and not self._stop_event.is_set():
                time.sleep(min(1.0, wait - waited))
                waited += 1.0
        self._running = False
        self.logger.info('监控已停止')

    # ---------------- 控制接口 ----------------
    def start(self):
        """启动后台监控线程（非阻塞）"""
        if self._running:
            self.logger.info('监控已在运行中')
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name='StockMonitor', daemon=True)
        self._thread.start()

    def pause(self):
        """暂停监控（线程保留）"""
        self._pause_event.set()
        self.logger.info('监控已暂停')

    def resume(self):
        """恢复监控"""
        self._pause_event.clear()
        self.logger.info('监控已恢复')

    def stop(self):
        """停止监控并回收线程"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 5)
        self._running = False

    @property
    def is_running(self):
        return self._running

    def run_once_sync(self):
        """同步执行一轮（供测试/单次运行使用，不启动线程）"""
        self._run_once()


def _parse_args(argv):
    """解析启动参数，返回 (watchlist, interval, once, interactive)"""
    watchlist = None
    interval = None
    once = 'once' in argv
    interactive = 'nointeractive' not in argv
    for arg in argv:
        if arg.startswith('interval='):
            try:
                interval = int(arg.split('=', 1)[1])
            except ValueError:
                pass
        elif arg.startswith('code='):
            watchlist = [c for c in arg.split('=', 1)[1].split(',') if c]
    return watchlist, interval, once, interactive


def main():
    watchlist, interval, once, interactive = _parse_args(sys.argv[1:])
    monitor = StockMonitor(watchlist=watchlist, interval=interval)

    if once:
        # 单轮同步执行，便于测试
        print('单轮监控模式（once），执行一轮后退出')
        monitor.run_once_sync()
        return

    monitor.start()

    if not interactive:
        # 非交互模式：主线程等待，直到 Ctrl+C
        print('监控后台运行中（nointeractive），按 Ctrl+C 停止')
        try:
            while monitor.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()
        return

    # 交互式控制台：主进程不被阻塞，可随时输入控制命令
    print('监控已启动。可用命令: pause / resume / stop / status')
    try:
        while monitor.is_running:
            cmd = input('monitor> ').strip().lower()
            if cmd == 'pause':
                monitor.pause()
            elif cmd == 'resume':
                monitor.resume()
            elif cmd == 'stop':
                monitor.stop()
                break
            elif cmd == 'status':
                print(f'运行中={monitor.is_running} 暂停={monitor._pause_event.is_set()} '
                      f'监控数={len(monitor.watchlist)} 间隔={monitor.interval}秒')
            elif cmd:
                print('未知命令，可用: pause / resume / stop / status')
    except (KeyboardInterrupt, EOFError):
        monitor.stop()


if __name__ == '__main__':
    main()
