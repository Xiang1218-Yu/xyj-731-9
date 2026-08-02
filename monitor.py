"""
实时行情监控与预警系统

功能：
  - 每 30 秒从通达信获取一次实时行情
  - 对监控列表中的股票持续运行选股策略
  - 当触发买入/卖出信号时，记录信号时间、价格、触发策略，生成预警日志
  - 支持启动 / 停止 / 暂停控制，监控在后台线程运行，不阻塞主进程

命令行用法：
  python monitor.py                  # 使用全部候选股票监控
  python monitor.py 000001 600030    # 只监控指定股票
  python monitor.py --interval 60    # 指定轮询间隔（秒），默认30
  python monitor.py --multi config.json       # 使用多因子策略引擎配置
  python monitor.py --sell-buy-func 策略2     # 指定卖出策略依赖的买入信号函数
  python monitor.py --sell-func 卖策略        # 指定卖出信号函数

多因子配置中可通过 sell_config 指定卖出策略：
  {
    "logic": "AND", "strategies": [...],
    "sell_config": {"buy_signal_func": "策略2", "sell_func": "卖策略"}
  }

编程用法：
  from monitor import StockMonitor
  monitor = StockMonitor(stocklist=['000001', '600030'])
  monitor.start()     # 后台启动，不阻塞
  monitor.pause()     # 暂停
  monitor.resume()    # 恢复
  monitor.stop()      # 停止
"""
import os
import sys
import time
import json
import threading
import logging
from datetime import datetime
from queue import Queue

import pandas as pd
from rich import print

import CeLue
import func
import user_config as ucfg
import strategy_engine

# 日志目录
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'monitor_logs')
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger():
    """配置预警日志记录器，同时输出到文件和控制台"""
    logger = logging.getLogger('stock_monitor')
    logger.setLevel(logging.INFO)
    # 避免重复添加 handler
    if logger.handlers:
        return logger

    log_file = os.path.join(LOG_DIR, f'alert_{time.strftime("%Y%m%d", time.localtime())}.log')
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class StockMonitor:
    """
    实时行情监控器。在后台守护线程中运行，不阻塞主进程。

    :param stocklist: list 股票代码列表，为 None 时使用全部候选股票
    :param interval: int 轮询间隔秒数，默认30
    :param engine_config: dict|None 多因子策略引擎配置，为 None 时使用默认策略1+策略2
    :param use_hs300_filter: bool 是否启用 HS300 大盘过滤
    """

    STATUS_STOPPED = 'stopped'
    STATUS_RUNNING = 'running'
    STATUS_PAUSED = 'paused'

    def __init__(self, stocklist=None, interval=30, engine_config=None, use_hs300_filter=True,
                 sell_buy_func='策略2', sell_func='卖策略'):
        """
        :param sell_buy_func: str 卖出策略所依赖的买入信号函数名（CeLue.py 中的函数名），
                                    多因子模式下应与引擎中的主买入策略一致，默认'策略2'
        :param sell_func: str 卖出信号函数名，默认'卖策略'
        """
        self.interval = interval
        self.engine_config = engine_config
        self.use_hs300_filter = use_hs300_filter
        self.sell_buy_func_name = sell_buy_func
        self.sell_func_name = sell_func
        self.logger = setup_logger()

        # 从 engine_config 中提取 sell_config 覆盖默认值
        if engine_config and 'sell_config' in engine_config:
            sc = engine_config['sell_config']
            self.sell_buy_func_name = sc.get('buy_signal_func', self.sell_buy_func_name)
            self.sell_func_name = sc.get('sell_func', self.sell_func_name)

        # 运行状态控制
        self._status = self.STATUS_STOPPED
        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # set 表示非暂停状态
        self._lock = threading.Lock()

        # 信号队列，外部可消费
        self.alert_queue = Queue()

        # 股票列表
        if stocklist is None:
            self.stocklist = self._make_default_stocklist()
        else:
            self.stocklist = list(stocklist)

        # 每只股票的上一次信号状态，用于检测信号边沿触发
        # {stockcode: {'buy': bool, 'sell': bool, 'strategies': list, 'price': float}}
        self._last_signals = {}

        # 多因子引擎（在工作线程中构建，避免 pickle 问题）
        self._engine = None
        self._sell_buy_func = None
        self._sell_func = None

        # HS300 信号
        self._hs300_signal = None

    def _make_default_stocklist(self):
        """生成默认候选股票列表（复用 xuangu.py 的筛选逻辑）"""
        from xuangu import make_stocklist
        try:
            return make_stocklist()
        except Exception as e:
            self.logger.warning(f'生成默认股票列表失败: {e}，使用 pickle 目录全部股票')
            return [i[:-4] for i in os.listdir(ucfg.tdx['pickle']) if i.endswith('.pkl')]

    def _build_engine(self):
        """在工作线程中构建策略引擎和卖出策略函数引用"""
        if self.engine_config is not None:
            self._engine = strategy_engine.build_engine_from_config(self.engine_config)
        else:
            self._engine = strategy_engine.build_default_engine()

        # 解析卖出策略所需的函数引用，确保多因子模式下卖出逻辑与买入策略一致
        self._sell_buy_func = getattr(CeLue, self.sell_buy_func_name, CeLue.策略2)
        self._sell_func = getattr(CeLue, self.sell_func_name, CeLue.卖策略)
        self.logger.info(f'卖出策略: 买入信号函数={self.sell_buy_func_name}, 卖出函数={self.sell_func_name}')

    def _update_hs300_signal(self):
        """更新 HS300 大盘信号"""
        try:
            df_hs300 = pd.read_csv(ucfg.tdx['csv_index'] + '/000300.csv',
                                  index_col=None, encoding='gbk', dtype={'code': str})
            df_hs300['date'] = pd.to_datetime(df_hs300['date'], format='%Y-%m-%d')
            df_hs300.set_index('date', drop=False, inplace=True)
            # 交易时段获取实时行情
            now = time.localtime()
            if '09:00:00' < time.strftime("%H:%M:%S", now) < '16:00:00' and 0 <= now.tm_wday <= 4:
                df_today = func.get_tdx_lastestquote((1, '000300'))
                df_hs300 = func.update_stockquote('000300', df_hs300, df_today)
            self._hs300_signal = CeLue.策略HS300(df_hs300)
        except Exception as e:
            self.logger.warning(f'获取 HS300 信号失败: {e}')
            self._hs300_signal = None

    def _get_realtime_quotes(self):
        """获取监控列表的实时行情"""
        try:
            return func.get_tdx_lastestquote(self.stocklist)
        except Exception as e:
            self.logger.error(f'获取实时行情失败: {e}')
            return None

    def _evaluate_stock(self, stockcode, df_today):
        """
        对单只股票执行策略评估。
        返回 dict: {'buy': bool, 'sell': bool, 'strategies': list, 'price': float, 'score': float}
        """
        pklfile = ucfg.tdx['pickle'] + os.sep + stockcode + '.pkl'
        try:
            df_stock = pd.read_pickle(pklfile)
        except FileNotFoundError:
            return None

        if df_today is not None:
            df_today_code = df_today.loc[df_today['code'] == stockcode]
            if len(df_today_code) > 0:
                df_stock = func.update_stockquote(stockcode, df_stock, df_today_code)

        df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')
        df_stock.set_index('date', drop=False, inplace=True)

        current_price = float(df_stock['close'].iat[-1])
        result = {
            'buy': False,
            'sell': False,
            'strategies': [],
            'price': current_price,
            'score': 0.0,
        }

        # 多因子引擎评估
        if self._engine is not None:
            context = {}
            if self._hs300_signal is not None:
                context['HS300_信号'] = self._hs300_signal
            try:
                engine_result = self._engine.evaluate(df_stock, **context)
                if engine_result.matched:
                    result['buy'] = True
                    result['strategies'] = engine_result.matched_strategies
                    result['score'] = engine_result.score
            except Exception as e:
                self.logger.debug(f'{stockcode} 策略引擎评估异常: {e}')

        # 卖出策略评估：使用配置的买入信号函数生成历史信号序列，再传给卖出函数
        # 多因子模式下 sell_buy_func 应与引擎主买入策略对应，确保买卖逻辑一致
        try:
            if self._sell_buy_func is not None and self._sell_func is not None:
                hs300 = self._hs300_signal if self._hs300_signal is not None \
                    else pd.Series(True, index=df_stock.index)
                # 买入信号序列由配置的函数生成
                buy_signal_series = self._sell_buy_func(df_stock, hs300)
                sell_signal = self._sell_func(df_stock, buy_signal_series)
                if isinstance(sell_signal, pd.Series) and len(sell_signal) > 0:
                    result['sell'] = bool(sell_signal.iat[-1])
                    if result['sell']:
                        sell_label = self.sell_func_name
                        if sell_label not in result['strategies']:
                            result['strategies'].append(sell_label)
        except Exception as e:
            self.logger.debug(f'{stockcode} 卖出策略评估异常: {e}')

        return result

    def _check_and_alert(self, stockcode, current_signal):
        """检测信号边沿变化，触发预警"""
        if current_signal is None:
            return

        last = self._last_signals.get(stockcode, {
            'buy': False, 'sell': False, 'strategies': [], 'price': 0.0
        })

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        price = current_signal['price']
        strategies = current_signal['strategies']

        # 买入信号：上一次未买入，本次买入
        if current_signal['buy'] and not last.get('buy', False):
            msg = (f'[BUY]  {stockcode}  价格:{price:.2f}  '
                   f'触发策略: {",".join(strategies) if strategies else "默认策略"}  '
                   f'评分:{current_signal["score"]:.2f}')
            self.logger.info(msg)
            alert = {
                'time': now_str, 'code': stockcode, 'type': 'BUY',
                'price': price, 'strategies': strategies,
                'score': current_signal['score'],
            }
            self.alert_queue.put(alert)

        # 卖出信号：上一次未卖出，本次卖出
        if current_signal['sell'] and not last.get('sell', False):
            msg = (f'[SELL] {stockcode}  价格:{price:.2f}  '
                   f'触发策略: {",".join(strategies) if strategies else "卖策略"}')
            self.logger.info(msg)
            alert = {
                'time': now_str, 'code': stockcode, 'type': 'SELL',
                'price': price, 'strategies': strategies,
                'score': current_signal['score'],
            }
            self.alert_queue.put(alert)

        self._last_signals[stockcode] = current_signal

    def _run_loop(self):
        """监控主循环（在后台线程中运行）"""
        self.logger.info(f'监控线程启动，共 {len(self.stocklist)} 只股票，轮询间隔 {self.interval} 秒')
        self._build_engine()

        # 首次更新 HS300 信号
        if self.use_hs300_filter:
            self._update_hs300_signal()

        while not self._stop_event.is_set():
            # 暂停等待
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            try:
                # 检查是否为交易时段
                now = time.localtime()
                is_trading_time = ('09:00:00' < time.strftime("%H:%M:%S", now) < '16:00:00'
                                   and 0 <= now.tm_wday <= 4)

                if not is_trading_time:
                    self.logger.debug('非交易时段，跳过本轮行情获取')
                else:
                    # 定期更新 HS300 信号（每10轮更新一次）
                    if self.use_hs300_filter and int(time.time()) % (self.interval * 10) < self.interval:
                        self._update_hs300_signal()

                    # 获取实时行情
                    df_today = self._get_realtime_quotes()

                    # 逐股评估
                    for stockcode in self.stocklist:
                        if self._stop_event.is_set():
                            break
                        self._pause_event.wait()
                        signal = self._evaluate_stock(stockcode, df_today)
                        self._check_and_alert(stockcode, signal)

            except Exception as e:
                self.logger.error(f'监控循环异常: {e}')

            # 等待下一轮，支持快速停止
            self._stop_event.wait(self.interval)

        self.logger.info('监控线程已停止')

    def start(self):
        """启动监控（后台守护线程，不阻塞主进程）"""
        with self._lock:
            if self._status == self.STATUS_RUNNING:
                self.logger.warning('监控已在运行中')
                return
            self._stop_event.clear()
            self._pause_event.set()
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name='StockMonitor')
            self._thread.start()
            self._status = self.STATUS_RUNNING
            self.logger.info('监控已启动')

    def stop(self):
        """停止监控"""
        with self._lock:
            if self._status == self.STATUS_STOPPED:
                return
            self._stop_event.set()
            self._pause_event.set()  # 解除暂停以让线程退出
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=self.interval + 5)
            self._status = self.STATUS_STOPPED
            self.logger.info('监控已停止')

    def pause(self):
        """暂停监控"""
        with self._lock:
            if self._status != self.STATUS_RUNNING:
                self.logger.warning(f'当前状态为 {self._status}，无法暂停')
                return
            self._pause_event.clear()
            self._status = self.STATUS_PAUSED
            self.logger.info('监控已暂停')

    def resume(self):
        """恢复监控"""
        with self._lock:
            if self._status != self.STATUS_PAUSED:
                self.logger.warning(f'当前状态为 {self._status}，无需恢复')
                return
            self._pause_event.set()
            self._status = self.STATUS_RUNNING
            self.logger.info('监控已恢复')

    def get_status(self):
        """获取当前监控状态"""
        return {
            'status': self._status,
            'stock_count': len(self.stocklist),
            'interval': self.interval,
            'tracked_signals': len(self._last_signals),
            'queue_size': self.alert_queue.qsize(),
        }

    def get_alerts(self, max_count=100):
        """从预警队列中取出所有待处理预警"""
        alerts = []
        while not self.alert_queue.empty() and len(alerts) < max_count:
            alerts.append(self.alert_queue.get())
        return alerts


def _parse_args(argv):
    """解析命令行参数"""
    stocks = []
    interval = 30
    engine_config_path = None
    sell_buy_func = '策略2'
    sell_func = '卖策略'
    i = 0
    while i < len(argv):
        if argv[i] == '--interval' and i + 1 < len(argv):
            interval = int(argv[i + 1])
            i += 2
        elif argv[i] == '--multi' and i + 1 < len(argv):
            engine_config_path = argv[i + 1]
            i += 2
        elif argv[i] == '--sell-buy-func' and i + 1 < len(argv):
            sell_buy_func = argv[i + 1]
            i += 2
        elif argv[i] == '--sell-func' and i + 1 < len(argv):
            sell_func = argv[i + 1]
            i += 2
        elif len(argv[i]) == 6 and argv[i].isdigit():
            stocks.append(argv[i])
            i += 1
        else:
            i += 1
    return stocks, interval, engine_config_path, sell_buy_func, sell_func


if __name__ == '__main__':
    stocks, interval, engine_config_path, sell_buy_func, sell_func = _parse_args(sys.argv[1:])

    engine_config = None
    if engine_config_path:
        if not os.path.exists(engine_config_path):
            print(f'[red]错误：配置文件 {engine_config_path} 不存在[/red]')
            sys.exit(1)
        with open(engine_config_path, 'r', encoding='utf-8') as f:
            engine_config = json.load(f)
        print(f'使用多因子策略配置: {engine_config_path}')

    print(f'[bold]股票实时行情监控系统[/bold]')
    print(f'轮询间隔: {interval} 秒')
    print(f'买入信号函数: {sell_buy_func} | 卖出函数: {sell_func}')
    if stocks:
        print(f'监控股票: {stocks}')
    else:
        print(f'未指定股票，将使用全部候选股票')

    monitor = StockMonitor(
        stocklist=stocks if stocks else None,
        interval=interval,
        engine_config=engine_config,
        sell_buy_func=sell_buy_func,
        sell_func=sell_func,
    )

    monitor.start()

    print('\n监控运行中。命令：')
    print('  p - 暂停    r - 恢复    s - 状态    q - 退出')
    print()

    try:
        while True:
            cmd = input().strip().lower()
            if cmd == 'p':
                monitor.pause()
            elif cmd == 'r':
                monitor.resume()
            elif cmd == 's':
                status = monitor.get_status()
                print(f'状态: {status["status"]} | 监控股票: {status["stock_count"]} | '
                      f'队列预警: {status["queue_size"]}')
            elif cmd == 'q':
                break
            else:
                print('未知命令。p=暂停 r=恢复 s=状态 q=退出')
    except (KeyboardInterrupt, EOFError):
        print('\n收到退出信号')
    finally:
        monitor.stop()
        print('程序退出')
