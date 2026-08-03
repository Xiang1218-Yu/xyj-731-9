#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
实时行情监控与预警系统（monitor）

功能：
1. 每 30 秒从通达信获取一次监控列表中股票的实时行情。
2. 对监控列表中的股票持续运行选股策略（复用 strategy_engine 多因子引擎，
   若未配置多因子则使用 CeLue.策略1(fast) 作为买入信号，CeLue.卖策略 作为卖出信号）。
3. 当股票触发买入/卖出信号时，记录：信号时间、股票代码、价格、触发策略，
   并写入预警日志（monitor_alerts.log）与 SQLite 数据库（monitor_alerts.db）。
4. 支持启动 / 停止 / 暂停控制，运行在独立后台线程，不阻塞主进程。
5. 同时提供命令行交互命令：status / pause / resume / stop / add <code> / del <code> / list / quit。

命令行启动参数：
  --watchlist 000001,600030    监控股票列表（逗号分隔，默认使用 xuangu.make_stocklist 的结果）
  --interval 30                行情刷新间隔（秒，默认 30）
  --multi-factor <json路径>    多因子策略 JSON 配置（可选，不传则使用内置买卖策略）
  --logfile monitor_alerts.log 预警日志文件路径
  --db monitor_alerts.db       预警 SQLite 数据库路径
  --once                       只运行一次（调试用，运行一轮策略后退出）

示例：
  python monitor.py --watchlist 000001,600030,300750 --interval 30
  python monitor.py --multi-factor my_strategy.json
"""

import os
import sys
import time
import argparse
import sqlite3
import threading
import datetime
import pandas as pd
from rich import print

import func
import user_config as ucfg
from strategy_engine import StrategyEngine, load_engine_from_json, CompositeStrategy, Strategy

# CeLue 模块依赖 talib，采用延迟导入，使 --help / --once 在无 talib 环境下也可运行
CeLue = None


def _ensure_celue():
    global CeLue
    if CeLue is None:
        import CeLue as _CeLue
        CeLue = _CeLue
    return CeLue


class AlertDatabase:
    """预警信息 SQLite 持久化。"""

    def __init__(self, db_path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        # check_same_thread=False 允许后台线程写入；通过 self._lock 保证线程安全
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_time TEXT    NOT NULL,
                    code        TEXT    NOT NULL,
                    signal_type TEXT    NOT NULL,
                    price       REAL,
                    strategies  TEXT,
                    score       REAL,
                    remark      TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_code ON alerts(code)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(signal_time)')
            conn.commit()

    def insert(self, signal_time, code, signal_type, price, strategies, score=None, remark=''):
        with self._lock, self._connect() as conn:
            conn.execute(
                'INSERT INTO alerts (signal_time, code, signal_type, price, strategies, score, remark) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (signal_time, code, signal_type, price,
                 ','.join(strategies) if isinstance(strategies, (list, tuple)) else str(strategies),
                 score, remark),
            )
            conn.commit()


class StockMonitor:
    """
    实时行情监控器。基于 threading.Event 实现暂停 / 停止控制，运行在独立线程。
    """

    def __init__(self, watchlist, engine=None, interval=30,
                 logfile='monitor_alerts.log', db_path='monitor_alerts.db'):
        self.watchlist = list(watchlist)
        self.engine = engine  # type: StrategyEngine
        self.interval = max(5, int(interval))
        self.logfile = logfile
        self.db = AlertDatabase(db_path)

        # 线程控制：_stop_event 控制退出，_pause_event 控制暂停（set=运行，clear=暂停）
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始为运行状态
        self._thread = None

        # 记录每只股票上一次的信号状态，用于边沿检测（只在信号新触发时报警）
        self._last_buy_state = {}   # code -> bool
        self._last_sell_state = {}  # code -> bool

        # 运行统计
        self._cycle_count = 0
        self._last_cycle_time = None
        self._alert_count = 0
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 控制接口
    # ------------------------------------------------------------------ #
    def start(self):
        """启动后台监控线程，不阻塞调用方。"""
        if self._thread is not None and self._thread.is_alive():
            print('[yellow]监控已在运行中[/yellow]')
            return
        self._stop_event.clear()
        self._pause_event.set()
        self._thread = threading.Thread(target=self._run, name='StockMonitor', daemon=True)
        self._thread.start()
        print(f'[green]监控线程已启动，刷新间隔 {self.interval} 秒，监控 {len(self.watchlist)} 只股票[/green]')

    def stop(self):
        """停止监控线程。"""
        self._stop_event.set()
        self._pause_event.set()  # 防止卡在暂停状态无法退出
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 5)
        print('[green]监控已停止[/green]')

    def pause(self):
        """暂停监控（保留已获取的数据，恢复后继续）。"""
        self._pause_event.clear()
        print('[yellow]监控已暂停[/yellow]')

    def resume(self):
        """恢复监控。"""
        self._pause_event.set()
        print('[green]监控已恢复[/green]')

    def is_running(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def is_paused(self):
        return self._pause_event.is_set() is False and self.is_running()

    def add_stock(self, code):
        with self._state_lock:
            if code not in self.watchlist:
                self.watchlist.append(code)
                print(f'[green]已添加 {code} 到监控列表（当前 {len(self.watchlist)} 只）[/green]')
            else:
                print(f'{code} 已在监控列表中')

    def del_stock(self, code):
        with self._state_lock:
            if code in self.watchlist:
                self.watchlist.remove(code)
                self._last_buy_state.pop(code, None)
                self._last_sell_state.pop(code, None)
                print(f'[green]已从监控列表移除 {code}（当前 {len(self.watchlist)} 只）[/green]')
            else:
                print(f'{code} 不在监控列表中')

    def status(self):
        state = '运行中' if self.is_running() else ('已暂停' if self.is_paused() else '已停止')
        with self._state_lock:
            print('─' * 50)
            print(f'状态: {state}')
            print(f'监控股票数: {len(self.watchlist)}')
            print(f'刷新间隔: {self.interval} 秒')
            print(f'已执行轮次: {self._cycle_count}')
            print(f'上次轮询: {self._last_cycle_time}')
            print(f'累计预警: {self._alert_count} 条')
            print(f'日志文件: {self.logfile}')
            print('─' * 50)

    def list_watchlist(self):
        with self._state_lock:
            print(f'监控列表（{len(self.watchlist)} 只）: {",".join(self.watchlist)}')

    # ------------------------------------------------------------------ #
    # 核心轮询逻辑
    # ------------------------------------------------------------------ #
    def _log_alert(self, code, signal_type, price, strategies, score=None):
        """写入预警日志和数据库。"""
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        strat_str = ','.join(strategies) if isinstance(strategies, (list, tuple)) else str(strategies)
        line = f'[{now}] {signal_type:4s} {code} 价格={price:.2f} 策略={strat_str}'
        if score is not None:
            line += f' 评分={score:.4f}'
        with open(self.logfile, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        color = 'red' if signal_type == 'BUY' else 'green'
        print(f'[{color}]{line}[/{color}]')
        self.db.insert(now, code, signal_type, price, strategies, score)
        with self._state_lock:
            self._alert_count += 1

    def _load_stock_df(self, stockcode, df_today):
        """读取本地 pkl 并合并实时行情。"""
        pklfile = ucfg.tdx['pickle'] + os.sep + stockcode + '.pkl'
        df_stock = pd.read_pickle(pklfile)
        if df_today is not None:
            df_today_code = df_today.loc[df_today['code'] == stockcode]
            if len(df_today_code) > 0:
                df_stock = func.update_stockquote(stockcode, df_stock, df_today_code)
        df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')
        df_stock.set_index('date', drop=False, inplace=True)
        return df_stock

    def _evaluate_signals(self, stockcode, df_stock, df_today_row):
        """
        计算单只股票的买入/卖出信号。
        返回 (buy_signal: bool, sell_signal: bool, buy_strategies: list, price, score)
        """
        price = float(df_stock['close'].iat[-1])

        if self.engine is not None and self.engine.root is not None:
            # 多因子引擎模式
            result = self.engine.evaluate_stock(df_stock)
            buy_signal = result['matched']
            buy_strategies = result['matched_strategies']
            score = result.get('score_normalized')
            # 卖出信号：若配置中包含卖出策略，通过 is_sell 标记识别；
            # 否则默认使用 CeLue.卖策略
            sell_signal = False
            sell_strategies = []
            sell_names = self.engine.get_sell_strategy_names()
            if sell_names:
                sell_signal = any(n in result['matched_strategies'] for n in sell_names)
                sell_strategies = [n for n in sell_names if n in result['matched_strategies']]
            else:
                try:
                    celue_mod = _ensure_celue()
                    celue2 = celue_mod.策略2(df_stock, self._hs300_signal)
                    sell_series = celue_mod.卖策略(df_stock, celue2)
                    sell_signal = bool(sell_series.iat[-1])
                    if sell_signal:
                        sell_strategies = ['卖策略']
                except Exception:
                    sell_signal = False
            return buy_signal, sell_signal, buy_strategies, sell_strategies, price, score

        # 默认模式：买入 = 策略1(fast)，卖出 = 卖策略
        celue_mod = _ensure_celue()
        try:
            buy_signal = bool(celue_mod.策略1(df_stock, mode='fast'))
        except Exception:
            buy_signal = False
        buy_strategies = ['策略1'] if buy_signal else []

        sell_signal = False
        sell_strategies = []
        try:
            celue2 = celue_mod.策略2(df_stock, self._hs300_signal)
            sell_series = celue_mod.卖策略(df_stock, celue2)
            sell_signal = bool(sell_series.iat[-1])
            if sell_signal:
                sell_strategies = ['卖策略']
        except Exception:
            sell_signal = False
        return buy_signal, sell_signal, buy_strategies, sell_strategies, price, None

    def _run_cycle(self):
        """执行一轮监控：拉取行情 → 计算策略 → 边沿检测 → 预警。"""
        with self._state_lock:
            watchlist = list(self.watchlist)

        # 获取实时行情
        try:
            df_today = func.get_tdx_lastestquote(watchlist)
        except Exception as e:
            print(f'[red]获取实时行情失败: {e}[/red]')
            return

        for code in watchlist:
            if self._stop_event.is_set():
                break
            try:
                df_stock = self._load_stock_df(code, df_today)
                buy, sell, buy_strats, sell_strats, price, score = self._evaluate_signals(
                    code, df_stock, df_today)

                # 边沿检测：只在信号从 False 变为 True 时预警
                prev_buy = self._last_buy_state.get(code, False)
                prev_sell = self._last_sell_state.get(code, False)
                if buy and not prev_buy:
                    self._log_alert(code, 'BUY', price, buy_strats, score)
                if sell and not prev_sell:
                    self._log_alert(code, 'SELL', price, sell_strats, score)
                self._last_buy_state[code] = buy
                self._last_sell_state[code] = sell
            except Exception as e:
                print(f'[red]处理 {code} 时出错: {e}[/red]')

    def _run(self):
        """后台线程主循环。"""
        # 预加载 HS300 信号
        try:
            df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv',
                                   index_col=None, encoding='gbk', dtype={'code': str})
            df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')
            df_hs300.set_index('date', drop=False, inplace=True)
            if '09:00:00' < time.strftime("%H:%M:%S", time.localtime()) < '16:00:00':
                df_today_hs = func.get_tdx_lastestquote((1, '000300'))
                df_hs300 = func.update_stockquote('000300', df_hs300, df_today_hs)
            self._hs300_signal = _ensure_celue().策略HS300(df_hs300)
        except Exception as e:
            print(f'[yellow]HS300 信号加载失败，使用全 True: {e}[/yellow]')
            self._hs300_signal = None

        print('[cyan]监控线程进入主循环[/cyan]')
        while not self._stop_event.is_set():
            # 暂停时阻塞等待
            self._pause_event.wait()
            if self._stop_event.is_set():
                break
            try:
                self._run_cycle()
            except Exception as e:
                print(f'[red]轮询异常: {e}[/red]')
            with self._state_lock:
                self._cycle_count += 1
                self._last_cycle_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 按 interval 秒等待，但支持快速响应 stop
            self._stop_event.wait(self.interval)


def parse_args():
    parser = argparse.ArgumentParser(
        description='实时行情监控与预警系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--watchlist', default=None,
                        help='监控股票列表，逗号分隔。不传则使用 xuangu.make_stocklist()')
    parser.add_argument('--interval', type=int, default=30,
                        help='行情刷新间隔（秒，默认 30，最小 5）')
    parser.add_argument('--multi-factor', dest='multi_factor', default=None,
                        help='多因子策略 JSON 配置文件路径（可选）')
    parser.add_argument('--logfile', default='monitor_alerts.log', help='预警日志文件路径')
    parser.add_argument('--db', default='monitor_alerts.db', help='预警 SQLite 数据库路径')
    parser.add_argument('--once', action='store_true', help='只运行一轮策略后退出（调试用）')
    return parser.parse_args()


def build_default_engine():
    """
    构建默认监控引擎：买入策略1(fast) AND 策略2，卖出使用 卖策略。
    监控模块以策略2作为买入信号更准确，因为策略2本身就是交易买点。
    """
    root = CompositeStrategy('默认监控策略', op='AND', children=[
        Strategy('快速筛选', '策略1', weight=1.0, strategy_kwargs={'mode': 'fast'}),
        Strategy('买入信号', '策略2', weight=2.0, needs_hs300=True),
    ])
    return StrategyEngine(root)


def interactive_shell(monitor):
    """提供简单的命令行交互界面。"""
    help_text = '''
可用命令:
  status              查看监控状态
  list                查看监控股票列表
  add <code>          添加股票到监控列表
  del <code>          从监控列表移除股票
  pause               暂停监控
  resume              恢复监控
  stop                停止监控
  quit / exit         退出程序（会自动停止监控）
  help                显示此帮助
'''.strip()
    print(help_text)
    while True:
        try:
            cmd = input('monitor> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            cmd = 'quit'
        if not cmd:
            continue
        parts = cmd.split()
        action = parts[0].lower()
        if action in ('quit', 'exit'):
            monitor.stop()
            print('程序退出')
            return
        elif action == 'help':
            print(help_text)
        elif action == 'status':
            monitor.status()
        elif action == 'list':
            monitor.list_watchlist()
        elif action == 'pause':
            monitor.pause()
        elif action == 'resume':
            monitor.resume()
        elif action == 'stop':
            monitor.stop()
        elif action == 'add' and len(parts) >= 2:
            monitor.add_stock(parts[1])
        elif action == 'del' and len(parts) >= 2:
            monitor.del_stock(parts[1])
        else:
            print(f'未知命令: {cmd}，输入 help 查看帮助')


def main():
    args = parse_args()

    # 构建监控列表
    if args.watchlist:
        watchlist = [c.strip() for c in args.watchlist.split(',') if c.strip()]
    else:
        try:
            from xuangu import make_stocklist
            watchlist = make_stocklist()
        except Exception as e:
            print(f'[red]无法生成股票列表: {e}，请使用 --watchlist 指定[/red]')
            sys.exit(1)

    # 构建策略引擎
    if args.multi_factor:
        print(f'[green]加载多因子策略配置: {args.multi_factor}[/green]')
        engine = load_engine_from_json(args.multi_factor)
    else:
        print('使用默认监控策略（策略1(fast) AND 策略2 作为买入，卖策略作为卖出）')
        engine = build_default_engine()

    monitor = StockMonitor(
        watchlist=watchlist,
        engine=engine,
        interval=args.interval,
        logfile=args.logfile,
        db_path=args.db,
    )

    if args.once:
        print('[cyan]单次运行模式，执行一轮策略后退出[/cyan]')
        monitor._run_cycle()
        monitor.status()
        return

    monitor.start()
    try:
        interactive_shell(monitor)
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == '__main__':
    main()
