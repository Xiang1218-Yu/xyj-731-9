import os
import sys
import copy
import time
import pickle
import sqlite3
import argparse
import pandas as pd
import numpy as np

import user_config as ucfg
from tqdm import tqdm
from rich import print as rprint

# ---------------------------------------------------------------------- #
# 命令行参数解析（支持不依赖 rqalpha 的独立分析模式）
# ---------------------------------------------------------------------- #
def _parse_huice_args():
    parser = argparse.ArgumentParser(
        description='策略回测与结果分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--analyze-only', dest='analyze_only', action='store_true',
                        help='仅基于信号数据(CSV/SQLite)计算夏普比率、最大回撤、胜率等指标，不运行 rqalpha')
    parser.add_argument('--db', default=None,
                        help='SQLite 信号数据库路径，默认使用 csv_gbbq/celue.db')
    parser.add_argument('--csv', default=None,
                        help='celue汇总.csv 路径，默认使用 csv_gbbq/celue汇总.csv')
    parser.add_argument('--start', default=None, help='分析起始日期 YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='分析结束日期 YYYY-MM-DD')
    parser.add_argument('--initial-cash', dest='initial_cash', type=float, default=1000000,
                        help='独立分析模式的初始资金（默认 1000000）')
    # 解析时忽略 rqalpha 自身可能传入的未知参数
    args, _ = parser.parse_known_args()
    return args


def load_signals_from_db(db_path, start_date=None, end_date=None):
    """从 SQLite 读取策略信号，返回 DataFrame。"""
    conn = sqlite3.connect(db_path)
    query = 'SELECT code, trade_date AS date, celue_buy, celue_sell, strategies, score, ' \
            'open, high, low, close, amount FROM signals'
    conditions = []
    params = []
    if start_date:
        conditions.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= ?")
        params.append(end_date)
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
    query += ' ORDER BY trade_date'
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df['celue_buy'] = df['celue_buy'].astype(bool)
    df['celue_sell'] = df['celue_sell'].astype(bool)
    return df


def load_signals_from_csv(csv_path):
    """从 celue汇总.csv 读取策略信号（兼容旧版格式）。"""
    df = pd.read_csv(csv_path, index_col=0, encoding='gbk', dtype={'code': str})
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    if 'celue_buy' in df.columns:
        df['celue_buy'] = df['celue_buy'].astype(bool)
    if 'celue_sell' in df.columns:
        df['celue_sell'] = df['celue_sell'].astype(bool)
    return df


def analyze_signals(df_signals, initial_cash=1000000):
    """
    基于买卖信号做独立的回测分析，输出夏普比率、最大回撤、胜率等统计指标。

    采用简化撮合规则：按日期顺序，买入信号次日开盘价买入（等权分配可用资金），
    卖出信号次日开盘价卖出全部持仓。不考虑涨跌停、手续费、滑点。
    返回包含统计指标的 dict。
    """
    if df_signals is None or len(df_signals) == 0:
        rprint('[red]无信号数据，无法分析[/red]')
        return {}

    df = df_signals.sort_values('date').reset_index(drop=True)

    # 需要每只股票的完整日线来获取次日开盘价，这里用信号表自身的 close 近似，
    # 若 open 列存在则优先用 open
    price_col = 'open' if 'open' in df.columns and df['open'].notna().any() else 'close'

    # 模拟组合
    cash = float(initial_cash)
    positions = {}  # code -> {'shares': int, 'cost_price': float}
    trade_records = []  # 每笔完整交易：buy_date, sell_date, code, buy_price, sell_price, return
    equity_curve = []

    all_dates = sorted(df['date'].unique())
    for i, current_date in enumerate(all_dates):
        day_df = df[df['date'] == current_date]
        next_date = all_dates[i + 1] if i + 1 < len(all_dates) else None

        # 先处理卖出
        for _, row in day_df.iterrows():
            code = row['code']
            if row.get('celue_sell', False) and code in positions:
                # 用当日 close 卖出（或次日 open，这里简化为当日 close）
                sell_price = float(row['close'])
                pos = positions.pop(code)
                proceeds = pos['shares'] * sell_price
                cash += proceeds
                trade_return = sell_price / pos['cost_price'] - 1
                trade_records.append({
                    'code': code,
                    'buy_date': pos['buy_date'],
                    'sell_date': current_date,
                    'buy_price': pos['cost_price'],
                    'sell_price': sell_price,
                    'return': trade_return,
                })

        # 再处理买入
        buy_today = day_df[day_df.get('celue_buy', False) == True]
        if len(buy_today) > 0 and next_date is not None:
            # 等权分配可用资金
            allocation = cash / max(1, len(buy_today))
            for _, row in buy_today.iterrows():
                code = row['code']
                if code in positions:
                    continue
                buy_price = float(row['close'])
                if buy_price <= 0:
                    continue
                shares = int(allocation // (buy_price * 100)) * 100  # A股按手
                if shares <= 0:
                    continue
                cost = shares * buy_price
                if cost > cash:
                    continue
                cash -= cost
                positions[code] = {'shares': shares, 'cost_price': buy_price, 'buy_date': current_date}

        # 记录当日权益
        market_value = 0.0
        for code, pos in positions.items():
            code_today = day_df[day_df['code'] == code]
            if len(code_today) > 0:
                market_value += pos['shares'] * float(code_today.iloc[0]['close'])
            else:
                market_value += pos['shares'] * pos['cost_price']
        total_equity = cash + market_value
        equity_curve.append({'date': current_date, 'equity': total_equity})

    # 收盘时仍持仓的，按最后一天收盘价平仓计算浮动盈亏（不计入胜率，但计入最终权益）
    df_equity = pd.DataFrame(equity_curve).set_index('date')

    # -------------------- 统计指标 -------------------- #
    # 日收益率
    df_equity['daily_return'] = df_equity['equity'].pct_change().fillna(0)

    total_return = df_equity['equity'].iloc[-1] / initial_cash - 1
    # 年化收益率（按252个交易日）
    n_days = len(df_equity)
    annualized_return = (1 + total_return) ** (252 / max(1, n_days)) - 1 if n_days > 0 else 0

    # 夏普比率（无风险利率取 0，日收益年化）
    if df_equity['daily_return'].std() > 0:
        sharpe = (df_equity['daily_return'].mean() / df_equity['daily_return'].std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # 最大回撤
    equity_series = df_equity['equity']
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 胜率
    df_trades = pd.DataFrame(trade_records)
    if len(df_trades) > 0:
        win_trades = len(df_trades[df_trades['return'] > 0])
        win_rate = win_trades / len(df_trades)
        avg_win = df_trades[df_trades['return'] > 0]['return'].mean() if win_trades > 0 else 0
        avg_loss = df_trades[df_trades['return'] <= 0]['return'].mean() if (len(df_trades) - win_trades) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_loss_ratio = 0.0

    stats = {
        '初始资金': initial_cash,
        '最终权益': round(float(df_equity['equity'].iloc[-1]), 2),
        '总收益率': round(float(total_return), 4),
        '年化收益率': round(float(annualized_return), 4),
        '夏普比率': round(float(sharpe), 4),
        '最大回撤': round(float(max_drawdown), 4),
        '交易笔数': len(df_trades),
        '盈利笔数': int(win_trades) if len(df_trades) > 0 else 0,
        '胜率': round(float(win_rate), 4),
        '平均盈利': round(float(avg_win), 4),
        '平均亏损': round(float(avg_loss), 4),
        '盈亏比': round(float(profit_loss_ratio), 4) if profit_loss_ratio != float('inf') else 'inf',
        '交易天数': n_days,
    }
    return stats


def print_analysis_report(stats):
    """格式化输出独立分析报告。"""
    if not stats:
        return
    rprint('=' * 60)
    rprint('[bold cyan]回测统计分析报告[/bold cyan]')
    rprint('=' * 60)
    rprint(f"初始资金:     {stats['初始资金']:>15,.2f}")
    rprint(f"最终权益:     {stats['最终权益']:>15,.2f}")
    rprint(f"总收益率:     {stats['总收益率']:>15.2%}")
    rprint(f"年化收益率:   {stats['年化收益率']:>15.2%}")
    rprint(f"夏普比率:     {stats['夏普比率']:>15.4f}")
    rprint(f"最大回撤:     {stats['最大回撤']:>15.2%}")
    rprint('-' * 60)
    rprint(f"交易笔数:     {stats['交易笔数']:>15d}")
    rprint(f"盈利笔数:     {stats['盈利笔数']:>15d}")
    rprint(f"胜率:         {stats['胜率']:>15.2%}")
    rprint(f"平均盈利:     {stats['平均盈利']:>15.2%}")
    rprint(f"平均亏损:     {stats['平均亏损']:>15.2%}")
    rprint(f"盈亏比:       {stats['盈亏比']:>15}")
    rprint(f"交易天数:     {stats['交易天数']:>15d}")
    rprint('=' * 60)


def _run_analyze_only():
    """独立分析模式入口：不启动 rqalpha，直接从信号数据计算统计指标。"""
    args = _parse_huice_args()
    # 优先使用 SQLite，其次 CSV
    db_path = args.db if args.db else (ucfg.tdx['csv_gbbq'] + os.sep + 'celue.db')
    csv_path = args.csv if args.csv else (ucfg.tdx['csv_gbbq'] + os.sep + 'celue汇总.csv')

    if os.path.exists(db_path):
        rprint(f'[green]从 SQLite 加载信号: {db_path}[/green]')
        df_signals = load_signals_from_db(db_path, args.start, args.end)
    elif os.path.exists(csv_path):
        rprint(f'[yellow]未找到 SQLite，从 CSV 加载信号: {csv_path}[/yellow]')
        df_signals = load_signals_from_csv(csv_path)
        if args.start:
            df_signals = df_signals[df_signals['date'] >= pd.to_datetime(args.start)]
        if args.end:
            df_signals = df_signals[df_signals['date'] <= pd.to_datetime(args.end)]
    else:
        rprint(f'[red]未找到信号数据库或CSV文件: {db_path} / {csv_path}[/red]')
        rprint('请先运行 celue_save.py 生成策略信号')
        sys.exit(1)

    stats = analyze_signals(df_signals, initial_cash=args.initial_cash)
    print_analysis_report(stats)
    return stats


# 仅在非独立分析模式下导入 rqalpha，避免无 rqalpha 环境时无法使用分析功能
_RQALPHA_AVAILABLE = False
if '--analyze-only' in sys.argv:
    if __name__ == '__main__':
        _run_analyze_only()
    # 模块被导入时不执行任何操作
else:
    try:
        import talib  # noqa: F401  保留与旧版一致的依赖导入
        from rqalpha.apis import *  # noqa: F401,F403
        from rqalpha import run_func  # noqa: F401
        _RQALPHA_AVAILABLE = True
    except ImportError:
        _RQALPHA_AVAILABLE = False

# 回测变量定义
start_date = "2013-01-01"  # 回测起始日期
end_date = "2022-12-31"  # 回测结束日期
stock_money = 10000000  # 股票账户初始资金
xiadan_percent = 0.1  # 设定买入总资产百分比的股票份额
xiadan_target_value = 100000  # 设定具体股票买入持有总金额
# 下单模式 买入总资产百分比的股票份额，或买入持有总金额的股票， 'order_percent' or 'order_target_value'
order_type = 'order_target_value'

rq_result_filename = "rq_result/" + time.strftime("%Y-%m-%d_%H%M%S", time.localtime()) + "+" + "start_date" + str(start_date)
rq_result_filename += "+" + order_type + "_" + (str(xiadan_percent) if order_type == 'order_percent' else str(xiadan_target_value))

os.mkdir("rq_result") if not os.path.exists("rq_result") else None
os.remove('temp.csv') if os.path.exists("temp.csv") else None


def update_stockcode(stockcode):
    if stockcode[0:1] == '6':
        stockcode = stockcode + ".XSHG"
    else:
        stockcode = stockcode + ".XSHE"
    return stockcode


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 在context中保存全局变量
    context.percent = xiadan_percent  # 设定买入比例
    context.target_value = xiadan_target_value  # 设定具体股票总买入市值
    context.order_type = order_type  # 下单模式

    df_celue = pd.read_csv(ucfg.tdx['csv_gbbq'] + os.sep + 'celue汇总.csv',
                           index_col=0, encoding='gbk', dtype={'code': str})
    df_celue['code'] = df_celue['code'].apply(lambda x: update_stockcode(x))  # 升级股票代码，匹配rqalpha
    df_celue['date'] = pd.to_datetime(df_celue['date'], format='%Y-%m-%d')  # 转为时间格式
    df_celue.set_index('date', drop=False, inplace=True)  # 时间为索引
    context.df_celue = df_celue


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    context.stock_pnl = pd.DataFrame()
    current_date = context.now.strftime('%Y-%m-%d')
    # 提取当天的df_celue
    if current_date in context.df_celue.index:
        context.df_today = context.df_celue.loc[[current_date]]
    else:
        context.df_today = None


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    if context.df_today is not None:
        for index, row in context.df_today.iterrows():
            # logger.info(index, row)

            # 检测是否停牌，停牌则交易单复制到下一个交易日
            if is_suspended(row['code']):
                # print(f"{row['code']} 停牌 跳过")
                row_new = copy.copy(row)
                # 获取下一个交易日日期，并赋值。新DF行附加到context.df_celue
                row_new['date'] = get_next_trading_date(context.now.strftime('%Y-%m-%d'), 1)
                row_new = pd.DataFrame(row_new).T.set_index('date', drop=False)
                context.df_celue = pd.concat([context.df_celue, row_new])
                continue

            # 获取当前投资组合中具体股票的数据
            cur_quantity = get_position(row['code']).quantity  # 该股持仓量
            cur_pnl = get_position(row['code']).pnl  # 该股持仓的累积盈亏

            # 卖出股票
            if row['celue_sell'] and cur_quantity > 0:
                order_result_obj = order_target_value(row['code'], 0)

                # order_result_obj.unfilled_quantity>0表示有未成交的委托股数，进行补单操作
                if order_result_obj.unfilled_quantity == 0:
                    # 委托单成交，计算收益率
                    # logger.info(f"SELL {row['code']}, 盈亏{round(get_position(row['code']).position_pnl, 2)}")

                    buy_price = context.df_celue.loc[(context.df_celue['code'] == row['code'])
                                                     & (context.df_celue['celue_buy'] == True)
                                                     & (context.df_celue['date'] < pd.to_datetime(context.now.strftime('%Y-%m-%d')))
                                                     ].iloc[-1].close
                    sell_price = context.df_today.loc[(context.df_today['code'] == row['code'])].iloc[-1].close
                    series = pd.Series(data={"trading_datetime": context.now,
                                             "order_book_id": row['code'],
                                             "side": "SELL",
                                             "盈亏金额": cur_pnl,
                                             "盈亏率": round(sell_price/buy_price-1, 4),
                                             })
                    context.stock_pnl = pd.concat([context.stock_pnl, series.to_frame().T], ignore_index=True)
                else:
                    # 委托单未成交
                    logger.info(f"{row['code']} {get_next_trading_date(context.now.strftime('%Y-%m-%d'))} 补单")
                    row_new = copy.copy(row)
                    # 获取下一个交易日日期，并赋值。新DF行附加到context.df_celue
                    row_new['date'] = get_next_trading_date(context.now.strftime('%Y-%m-%d'), 1)
                    row_new = pd.DataFrame(row_new).T.set_index('date', drop=False)
                    context.df_celue = pd.concat([context.df_celue, row_new])
                    # 根据日期删除有隐患，可能删除当日所有记录。不删程序也不影响
                    # context.df_celue.drop(
                    #     context.df_celue.loc[(context.df_celue['date'] == row['date'])
                    #                          & (context.df_celue['code'] == row['code'])].index,
                    #     inplace=True,
                    # )

            # 买入股票
            if row['celue_buy'] and cur_quantity == 0:
                if context.order_type == 'order_percent':
                    # 买入/卖出证券以自动调整该证券的仓位到占有一个目标价值。
                    # 加仓时，percent 代表证券已有持仓的价值加上即将花费的现金（包含税费）的总值占当前投资组合总价值的比例。
                    # 减仓时，percent 代表证券将被调整到的目标价至占当前投资组合总价值的比例。
                    order_result_obj = order_percent(row['code'], context.percent)
                    # logger.info(f"BUY {row['code']}")

                elif context.order_type == 'order_target_value':
                    # 买入 / 卖出并且自动调整该证券的仓位到一个目标价值。
                    # 加仓时，cash_amount代表现有持仓的价值加上即将花费（包含税费）的现金的总价值。
                    # 减仓时，cash_amount代表调整仓位的目标价至。
                    # 需要注意，如果资金不足，该API将不会创建发送订单。
                    order_result_obj = order_target_value(row['code'], context.target_value)
                    # logger.info(f"BUY {row['code']}")

                # 委托单成交状态判断处理
                # 订单被拒单：下单量为0 的情况。由于没有可用资金导致的，返回的order_result_obj是None类型，和其他情况不一样
                if order_result_obj is None:
                    string = f'净值{context.portfolio.total_value:>.2f} '
                    string += f'可用{context.portfolio.cash:>.2f} '
                    string += f'市值{context.portfolio.market_value:>.2f} '
                    # string += f'收益{context.portfolio.total_returns:>.2%} '
                    string += f'持股{len(context.portfolio.positions):>d} '
                    logger.info(string)
                # order_result_obj.unfilled_quantity>0表示有未成交的委托股数，进行补单操作
                elif order_result_obj.unfilled_quantity > 0:
                    logger.info(f"{row['code']} {get_next_trading_date(context.now.strftime('%Y-%m-%d'))} 补单")
                    row_new = copy.copy(row)
                    # 获取下一个交易日日期，并赋值。新DF行附加到context.df_celue
                    row_new['date'] = get_next_trading_date(context.now.strftime('%Y-%m-%d'), 1)
                    row_new = pd.DataFrame(row_new).T.set_index('date', drop=False)
                    context.df_celue = pd.concat([context.df_celue, row_new])
                # 订单成功完成
                else:
                    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    string = f'净值{context.portfolio.total_value:>.2f} '
    string += f'可用{context.portfolio.cash:>.2f} '
    string += f'市值{context.portfolio.market_value:>.2f} '
    # string += f'收益{context.portfolio.total_returns:>.2%} '
    string += f'持股{len(context.portfolio.positions):>d} '
    # logger.info(string)

    if len(context.stock_pnl) > 0:
        if os.path.exists('temp.csv'):
            context.stock_pnl.to_csv('temp.csv', encoding='gbk', mode='a', header=False)  # 附加数据，无标题行
        else:
            context.stock_pnl.to_csv('temp.csv', encoding='gbk', header=True)


__config__ = {
    "base": {
        # 回测起始日期
        "start_date": start_date,
        "end_date": end_date,
        # 数据源所存储的文件路径
        "data_bundle_path": os.path.expanduser("~/.rqalpha/bundle"),
        "strategy_file": "huice.py",
        # 目前支持 `1d` (日线回测) 和 `1m` (分钟线回测)，如果要进行分钟线，请注意是否拥有对应的数据源，目前开源版本是不提供对应的数据源的。
        "frequency": "1d",
        # 启用的回测引擎，目前支持 current_bar (当前Bar收盘价撮合) 和 next_bar (下一个Bar开盘价撮合)
        "matching_type": "current_bar",
        # 运行类型，`b` 为回测，`p` 为模拟交易, `r` 为实盘交易。
        "run_type": "b",
        # 设置策略可交易品种，目前支持 `stock` (股票账户)、`future` (期货账户)，您也可以自行扩展
        "accounts": {
            # 如果想设置使用某个账户，只需要增加对应的初始资金即可
            "stock": stock_money,
        },
        # 设置初始仓位
        "init_positions": {}
    },
    "extra": {
        # 选择日期的输出等级，有 `verbose` | `info` | `warning` | `error` 等选项，您可以通过设置 `verbose` 来查看最详细的日志，
        "log_level": "info",
    },

    "mod": {
        "sys_analyser": {
            "enabled": True,
            "benchmark": "000300.XSHG",
            # "plot": True,
            'plot_save_file': rq_result_filename + ".png",
            "output_file": rq_result_filename + ".pkl",
            # "report_save_path": "rq_result.csv",
        },
        # 策略运行过程中显示的进度条的控制
        "sys_progress": {
            "enabled": False,
            "show": True,
        },
    },
}

start_time = f'程序开始时间：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'

# 仅在 rqalpha 可用且非独立分析模式下执行回测
if _RQALPHA_AVAILABLE and __name__ == '__main__':
    # 使用 run_func 函数来运行策略
    # 此种模式下，您只需要在当前环境下定义策略函数，并传入指定运行的函数，即可运行策略。
    # 如果你的函数命名是按照 API 规范来，则可以直接按照以下方式来运行
    run_func(**globals())
    end_time = f'程序结束时间：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'

    # RQAlpha可以输出一个 pickle 文件，里面为一个 dict 。keys 包括
    # summary 回测摘要
    # stock_portfolios 股票帐号的市值
    # future_portfolios 期货帐号的市值
    # total_portfolios 总账号的的市值
    # benchmark_portfolios 基准帐号的市值
    # stock_positions 股票持仓
    # future_positions 期货仓位
    # benchmark_positions 基准仓位
    # trades 交易详情（交割单）
    # plots 调用plot画图时，记录的值
    result_dict = pd.read_pickle(rq_result_filename + ".pkl")

    # 给rq_result.pkl的交割单添加个股盈亏和收益率统计
    df_trades = result_dict['trades']
    try:
        df_temp = pd.read_csv('temp.csv', index_col=0, encoding='gbk')
        if 'trading_datetime' in df_temp.columns:
            df_temp = df_temp.set_index('trading_datetime')
        df_temp.index.name = 'datetime'
        # 避免列名冲突，重命名列
        df_temp = df_temp.rename(columns=lambda x: x + '_temp' if x in df_trades.columns else x)
        df_trades = pd.merge(df_trades, df_temp, left_index=True, right_index=True, how='left')
        result_dict['trades'] = df_trades
    except FileNotFoundError:
        print("temp.csv 不存在，跳过合并")
    except Exception as e:
        print(f"合并交割单数据出错: {e}")
    with open(rq_result_filename+".pkl", 'wb') as fobj:
        pickle.dump(result_dict, fobj)
    os.remove('temp.csv') if os.path.exists("temp.csv") else None


    rprint(result_dict["summary"])
    rprint(start_time)
    rprint(end_time)

    # -------------------- 新增：夏普比率、胜率等补充统计 -------------------- #
    # 从总账户净值序列计算夏普比率（rqalpha summary 中不一定包含）
    _extra_stats = {}
    try:
        if 'total_portfolios' in result_dict:
            _equity = result_dict['total_portfolios'].astype(float)
            _daily_ret = _equity.pct_change().dropna()
            if len(_daily_ret) > 1 and _daily_ret.std() > 0:
                _extra_stats['夏普比率'] = (_daily_ret.mean() / _daily_ret.std()) * np.sqrt(252)
            else:
                _extra_stats['夏普比率'] = 0.0
        elif 'sharpe' in result_dict['summary']:
            _extra_stats['夏普比率'] = result_dict['summary']['sharpe']
        else:
            _extra_stats['夏普比率'] = None
    except Exception as _e:
        _extra_stats['夏普比率'] = None

    # 从交割单计算胜率（基于平仓盈亏）
    try:
        _df_trades_all = result_dict.get('trades', pd.DataFrame())
        if len(_df_trades_all) > 0 and 'side' in _df_trades_all.columns:
            # rqalpha 的 trades 包含 buy/sell 两侧，用 last_quantity 或 position_pnl 判断
            if 'position_pnl' in _df_trades_all.columns:
                _sell_trades = _df_trades_all[_df_trades_all['side'].astype(str).str.upper() == 'SELL']
                _closed = _sell_trades[_sell_trades['last_quantity'] == 0] if 'last_quantity' in _sell_trades.columns else _sell_trades
                if len(_closed) > 0:
                    _pnl = pd.to_numeric(_closed['position_pnl'], errors='coerce').dropna()
                    _wins = (_pnl > 0).sum()
                    _extra_stats['平仓笔数'] = int(len(_pnl))
                    _extra_stats['盈利笔数'] = int(_wins)
                    _extra_stats['胜率'] = round(float(_wins / len(_pnl)), 4) if len(_pnl) > 0 else 0.0
                    _extra_stats['平均盈亏'] = round(float(_pnl.mean()), 2)
    except Exception as _e:
        pass

    rprint(
        f"回测起点 {result_dict['summary']['start_date']}"
        f"\n回测终点 {result_dict['summary']['end_date']}"
        f"\n回测收益 {result_dict['summary']['total_returns']:>.2%}\t年化收益 {result_dict['summary']['annualized_returns']:>.2%}"
        f"\t基准收益 {result_dict['summary']['benchmark_total_returns']:>.2%}\t基准年化 {result_dict['summary']['benchmark_annualized_returns']:>.2%}"
        f"\t最大回撤 {result_dict['summary']['max_drawdown']:>.2%}")
    if _extra_stats.get('夏普比率') is not None:
        rprint(f"夏普比率 {_extra_stats['夏普比率']:>.4f}")
    if '胜率' in _extra_stats:
        rprint(f"平仓笔数 {_extra_stats['平仓笔数']:>d}\t盈利笔数 {_extra_stats['盈利笔数']:>d}"
               f"\t胜率 {_extra_stats['胜率']:>.2%}\t平均盈亏 {_extra_stats['平均盈亏']:>.2f}")
    rprint("打开程序文件夹下的rq_result.png查看收益走势图")
    rprint("提示: 也可运行 python huice.py --analyze-only 直接基于 celue.db/celue汇总.csv 输出完整统计指标")
