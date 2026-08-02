import os
import sys
import copy
import time
import sqlite3
import pickle
import talib
import pandas as pd
import numpy as np

import user_config as ucfg
from rqalpha.apis import *
from rqalpha import run_func
from tqdm import tqdm
from rich import print as rprint

# 回测变量定义
start_date = "2013-01-01"  # 回测起始日期
end_date = "2022-12-31"  # 回测结束日期
stock_money = 10000000  # 股票账户初始资金
xiadan_percent = 0.1  # 设定买入总资产百分比的股票份额
xiadan_target_value = 100000  # 设定具体股票买入持有总金额
# 下单模式 买入总资产百分比的股票份额，或买入持有总金额的股票， 'order_percent' or 'order_target_value'
order_type = 'order_target_value'

# 数据源配置：--db 使用 SQLite 数据库，默认使用 CSV
_use_sqlite = '--db' in sys.argv
_db_path = 'strategy_signals.db'
if _use_sqlite:
    _idx = sys.argv.index('--db')
    if _idx + 1 < len(sys.argv) and not sys.argv[_idx + 1].startswith('--'):
        _db_path = sys.argv[_idx + 1]

# 仅分析模式：--analyze 跳过回测直接分析已有结果
_analyze_only = '--analyze' in sys.argv
_analyze_file = None
if _analyze_only:
    _idx = sys.argv.index('--analyze')
    if _idx + 1 < len(sys.argv) and not sys.argv[_idx + 1].startswith('--'):
        _analyze_file = sys.argv[_idx + 1]

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


def load_celue_from_sqlite(db_path, start_date=None, end_date=None):
    """
    从 SQLite 数据库加载策略信号，返回与 celue汇总.csv 格式兼容的 DataFrame。
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT code, date, close, celue_buy, celue_sell, strategy, open, high, low FROM strategy_signals"
    conditions = []
    params = []
    if start_date:
        conditions.append("date >= ?")
        params.append(str(start_date))
    if end_date:
        conditions.append("date <= ?")
        params.append(str(end_date))
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    df['code'] = df['code'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['celue_buy'] = df['celue_buy'].astype(bool)
    df['celue_sell'] = df['celue_sell'].astype(bool)
    return df


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 在context中保存全局变量
    context.percent = xiadan_percent  # 设定买入比例
    context.target_value = xiadan_target_value  # 设定具体股票总买入市值
    context.order_type = order_type  # 下单模式

    # 支持从 SQLite 数据库或 CSV 文件加载策略信号
    if _use_sqlite:
        rprint(f'[blue]从 SQLite 数据库加载策略信号: {_db_path}[/blue]')
        df_celue = load_celue_from_sqlite(_db_path, start_date=start_date, end_date=end_date)
    else:
        df_celue = pd.read_csv(ucfg.tdx['csv_gbbq'] + os.sep + 'celue汇总.csv',
                               index_col=0, encoding='gbk', dtype={'code': str})
        df_celue['date'] = pd.to_datetime(df_celue['date'], format='%Y-%m-%d')  # 转为时间格式

    df_celue['code'] = df_celue['code'].apply(lambda x: update_stockcode(x))  # 升级股票代码，匹配rqalpha
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


# ============================================================
# 回测结果统计分析函数
# ============================================================

def calculate_sharpe_ratio(portfolio_series, risk_free_rate=0.03):
    """
    计算夏普比率。
    :param portfolio_series: pd.Series 每日组合总市值序列
    :param risk_free_rate: float 无风险年利率，默认3%
    :return: float 年化夏普比率
    """
    if portfolio_series is None or len(portfolio_series) < 2:
        return 0.0
    daily_returns = portfolio_series.pct_change().dropna()
    if len(daily_returns) == 0 or daily_returns.std() == 0:
        return 0.0
    # 中国市场一年约244个交易日
    sharpe = (daily_returns.mean() * 244 - risk_free_rate) / (daily_returns.std() * np.sqrt(244))
    return float(sharpe)


def calculate_max_drawdown(portfolio_series):
    """
    计算最大回撤。
    :param portfolio_series: pd.Series 每日组合总市值序列
    :return: (max_drawdown, peak_date, trough_date)
    """
    if portfolio_series is None or len(portfolio_series) < 2:
        return 0.0, None, None
    cumulative_max = portfolio_series.cummax()
    drawdown = (portfolio_series - cumulative_max) / cumulative_max
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = portfolio_series.loc[:trough_date].idxmax() if trough_date is not None else None
    return float(max_dd), peak_date, trough_date


def calculate_win_rate(df_trades):
    """
    计算胜率、盈亏比等交易统计指标。
    :param df_trades: pd.DataFrame 交易交割单
    :return: dict 统计指标
    """
    stats = {
        'total_trades': 0,
        'win_trades': 0,
        'loss_trades': 0,
        'win_rate': 0.0,
        'avg_profit': 0.0,
        'avg_loss': 0.0,
        'profit_loss_ratio': 0.0,
        'max_profit': 0.0,
        'max_loss': 0.0,
    }
    if df_trades is None or len(df_trades) == 0:
        return stats

    pnl_col = None
    for col in ['盈亏率', '盈亏率_temp', 'pnl_ratio']:
        if col in df_trades.columns:
            pnl_col = col
            break

    if pnl_col is None:
        return stats

    pnl_series = pd.to_numeric(df_trades[pnl_col], errors='coerce').dropna()
    if 'side' in df_trades.columns:
        sell_mask = df_trades['side'].astype(str).str.upper() == 'SELL'
        pnl_series = pd.to_numeric(df_trades.loc[sell_mask, pnl_col], errors='coerce').dropna()

    if len(pnl_series) == 0:
        return stats

    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]

    stats['total_trades'] = len(pnl_series)
    stats['win_trades'] = len(wins)
    stats['loss_trades'] = len(losses)
    stats['win_rate'] = len(wins) / len(pnl_series) if len(pnl_series) > 0 else 0.0
    stats['avg_profit'] = float(wins.mean()) if len(wins) > 0 else 0.0
    stats['avg_loss'] = float(losses.mean()) if len(losses) > 0 else 0.0
    if stats['avg_loss'] != 0:
        stats['profit_loss_ratio'] = abs(stats['avg_profit'] / stats['avg_loss'])
    stats['max_profit'] = float(pnl_series.max())
    stats['max_loss'] = float(pnl_series.min())
    return stats


def calculate_annualized_return(portfolio_series, trading_days=244):
    """计算年化收益率"""
    if portfolio_series is None or len(portfolio_series) < 2:
        return 0.0
    total_return = portfolio_series.iat[-1] / portfolio_series.iat[0] - 1
    years = len(portfolio_series) / trading_days
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def calculate_volatility(portfolio_series, trading_days=244):
    """计算年化波动率"""
    if portfolio_series is None or len(portfolio_series) < 2:
        return 0.0
    daily_returns = portfolio_series.pct_change().dropna()
    return float(daily_returns.std() * np.sqrt(trading_days))


def analyze_backtest_results(result_dict):
    """
    综合分析回测结果，输出夏普比率、最大回撤、胜率等统计指标。
    :param result_dict: dict rqalpha 回测结果字典
    :return: dict 增强后的统计指标
    """
    summary = result_dict.get('summary', {})
    portfolios = result_dict.get('stock_portfolios', None)

    portfolio_series = None
    if portfolios is not None:
        if isinstance(portfolios, pd.DataFrame):
            if 'total_value' in portfolios.columns:
                portfolio_series = portfolios['total_value']
            elif 'unit_net_value' in portfolios.columns:
                portfolio_series = portfolios['unit_net_value']
        elif isinstance(portfolios, pd.Series):
            portfolio_series = portfolios

    sharpe = calculate_sharpe_ratio(portfolio_series)
    max_dd, peak_date, trough_date = calculate_max_drawdown(portfolio_series)
    annual_ret = calculate_annualized_return(portfolio_series)
    volatility = calculate_volatility(portfolio_series)

    df_trades = result_dict.get('trades', None)
    trade_stats = calculate_win_rate(df_trades)

    enhanced = {
        'sharpe_ratio': sharpe,
        'max_drawdown_calc': max_dd,
        'max_drawdown_peak': str(peak_date) if peak_date is not None else '',
        'max_drawdown_trough': str(trough_date) if trough_date is not None else '',
        'annualized_return_calc': annual_ret,
        'volatility': volatility,
        **trade_stats,
    }
    return enhanced


def print_enhanced_summary(result_dict, enhanced_stats):
    """打印增强版回测统计摘要"""
    summary = result_dict.get('summary', {})
    rprint('\n' + '=' * 60)
    rprint('[bold blue]回测结果统计分析报告[/bold blue]')
    rprint('=' * 60)

    rprint(f"[bold]基础收益指标:[/bold]")
    rprint(f"  回测区间      : {summary.get('start_date', 'N/A')} ~ {summary.get('end_date', 'N/A')}")
    rprint(f"  总收益率      : {summary.get('total_returns', 0):>.2%}")
    rprint(f"  年化收益率    : {summary.get('annualized_returns', enhanced_stats.get('annualized_return_calc', 0)):>.2%}")
    rprint(f"  基准收益率    : {summary.get('benchmark_total_returns', 0):>.2%}")
    rprint(f"  基准年化      : {summary.get('benchmark_annualized_returns', 0):>.2%}")
    rprint(f"  超额收益      : {summary.get('total_returns', 0) - summary.get('benchmark_total_returns', 0):>.2%}")

    rprint(f"\n[bold]风险指标:[/bold]")
    rprint(f"  夏普比率      : {enhanced_stats['sharpe_ratio']:>.3f}")
    rprint(f"  最大回撤      : {summary.get('max_drawdown', enhanced_stats['max_drawdown_calc']):>.2%}")
    if enhanced_stats['max_drawdown_peak']:
        rprint(f"  回撤区间      : {enhanced_stats['max_drawdown_peak']} ~ {enhanced_stats['max_drawdown_trough']}")
    rprint(f"  年化波动率    : {enhanced_stats['volatility']:>.2%}")

    rprint(f"\n[bold]交易统计:[/bold]")
    rprint(f"  总交易次数    : {enhanced_stats['total_trades']}")
    rprint(f"  盈利次数      : {enhanced_stats['win_trades']}")
    rprint(f"  亏损次数      : {enhanced_stats['loss_trades']}")
    rprint(f"  胜率          : {enhanced_stats['win_rate']:>.2%}")
    rprint(f"  平均盈利      : {enhanced_stats['avg_profit']:>.2%}")
    rprint(f"  平均亏损      : {enhanced_stats['avg_loss']:>.2%}")
    rprint(f"  盈亏比        : {enhanced_stats['profit_loss_ratio']:>.3f}")
    rprint(f"  单笔最大盈利  : {enhanced_stats['max_profit']:>.2%}")
    rprint(f"  单笔最大亏损  : {enhanced_stats['max_loss']:>.2%}")
    rprint('=' * 60 + '\n')


start_time = f'程序开始时间：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'

# ============================================================
# 主执行流程：支持 --analyze 仅分析模式
# ============================================================
if _analyze_only:
    # 仅分析模式：跳过回测，直接分析已有结果
    if _analyze_file is None:
        rq_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rq_result')
        pkl_files = sorted([f for f in os.listdir(rq_dir) if f.endswith('.pkl')], reverse=True)
        if not pkl_files:
            rprint('[red]错误：未找到任何回测结果 pkl 文件[/red]')
            sys.exit(1)
        _analyze_file = os.path.join(rq_dir, pkl_files[0])
    rprint(f'[blue]仅分析模式，加载结果文件: {_analyze_file}[/blue]')
    result_dict = pd.read_pickle(_analyze_file)
    enhanced_stats = analyze_backtest_results(result_dict)
    rprint(result_dict.get('summary', {}))
    print_enhanced_summary(result_dict, enhanced_stats)
else:
    # 正常回测流程
    run_func(**globals())
    end_time = f'程序结束时间：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'

    # RQAlpha可以输出一个 pickle 文件，里面为一个 dict
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

    # 计算增强统计指标并保存回结果文件
    enhanced_stats = analyze_backtest_results(result_dict)
    result_dict['enhanced_stats'] = enhanced_stats
    with open(rq_result_filename + ".pkl", 'wb') as fobj:
        pickle.dump(result_dict, fobj)
    os.remove('temp.csv') if os.path.exists("temp.csv") else None

    # 输出增强版统计报告
    print_enhanced_summary(result_dict, enhanced_stats)
    rprint(start_time)
    rprint(end_time)
    rprint(
        f"回测起点 {result_dict['summary']['start_date']}"
        f"\n回测终点 {result_dict['summary']['end_date']}"
        f"\n回测收益 {result_dict['summary']['total_returns']:>.2%}\t年化收益 {result_dict['summary']['annualized_returns']:>.2%}"
        f"\t基准收益 {result_dict['summary']['benchmark_total_returns']:>.2%}\t基准年化 {result_dict['summary']['benchmark_annualized_returns']:>.2%}"
        f"\t最大回撤 {result_dict['summary']['max_drawdown']:>.2%}"
        f"\n打开程序文件夹下的rq_result.png查看收益走势图")
