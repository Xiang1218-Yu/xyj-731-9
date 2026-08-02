import os
import copy
import sqlite3
import sys
import time
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


def read_celue_signals():
    """
    读取回测策略信号。默认优先读取SQLite数据库 celue.db（celue_save.py生成），
    数据库不存在或附带命令行参数 csv 时，读取旧版 celue汇总.csv
    :return: DF格式，策略买卖信号汇总
    """
    db_path = ucfg.tdx['csv_gbbq'] + os.sep + 'celue.db'
    if 'csv' not in sys.argv[1:] and os.path.exists(db_path):
        # 从SQLite数据库读取（新持久化格式，含触发策略、前复权价格字段）
        conn = sqlite3.connect(db_path)
        try:
            df_celue = pd.read_sql('SELECT * FROM celue_signals', conn, dtype={'code': str})
        finally:
            conn.close()
        df_celue['celue_buy'] = df_celue['celue_buy'].astype(bool)
        df_celue['celue_sell'] = df_celue['celue_sell'].astype(bool)
    else:
        df_celue = pd.read_csv(ucfg.tdx['csv_gbbq'] + os.sep + 'celue汇总.csv',
                               index_col=0, encoding='gbk', dtype={'code': str})
    df_celue['code'] = df_celue['code'].apply(lambda x: update_stockcode(x))  # 升级股票代码，匹配rqalpha
    df_celue['date'] = pd.to_datetime(df_celue['date'], format='%Y-%m-%d')  # 转为时间格式
    df_celue.set_index('date', drop=False, inplace=True)  # 时间为索引
    return df_celue


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 在context中保存全局变量
    context.percent = xiadan_percent  # 设定买入比例
    context.target_value = xiadan_target_value  # 设定具体股票总买入市值
    context.order_type = order_type  # 下单模式

    context.df_celue = read_celue_signals()


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

# ==================== 回测结果分析功能 ====================
RISK_FREE_RATE = 0.03  # 无风险年化利率，夏普比率计算使用


def extract_trade_returns(result_dict):
    """
    从交割单(trades)提取每笔已平仓交易的收益率序列。
    优先使用after_trading合并的"盈亏率"列；不存在时按先进先出(FIFO)配对买卖单计算
    :return: pd.Series 每笔交易的收益率
    """
    df_trades = result_dict.get('trades')
    if df_trades is None or len(df_trades) == 0:
        return pd.Series(dtype=float)
    if '盈亏率' in df_trades.columns:
        ret = df_trades.loc[df_trades['side'].astype(str).str.upper().str.endswith('SELL'), '盈亏率'].dropna()
        if len(ret) > 0:
            return ret.astype(float)
    # 无盈亏率列：按股票FIFO配对买卖单，逐笔计算收益率
    # 兼容rqalpha不同版本交割单字段：4.x为price/quantity，5.x+为last_price/last_quantity
    col_price = 'price' if 'price' in df_trades.columns else 'last_price'
    col_qty = 'quantity' if 'quantity' in df_trades.columns else 'last_quantity'
    if col_price not in df_trades.columns or col_qty not in df_trades.columns:
        return pd.Series(dtype=float)
    returns = []
    buy_queues = {}  # 每只股票未平仓的买入队列 [[价格, 数量], ...]
    for _, trade in df_trades.sort_index().iterrows():
        book = trade['order_book_id']
        side = str(trade['side']).upper()
        queue = buy_queues.setdefault(book, [])
        if side.endswith('BUY'):
            queue.append([trade[col_price], trade[col_qty]])
        elif side.endswith('SELL') and queue:
            remain = trade[col_qty]
            while remain > 0 and queue:
                lot = queue[0]
                matched = min(lot[1], remain)
                returns.append(trade[col_price] / lot[0] - 1)
                lot[1] -= matched
                remain -= matched
                if lot[1] <= 0:
                    queue.pop(0)
    return pd.Series(returns, dtype=float)


def analyze_result(result_dict, rf=RISK_FREE_RATE):
    """
    回测结果统计分析，输出夏普比率、最大回撤、胜率等指标
    :param result_dict: rqalpha输出的pkl内容dict（含total_portfolios/trades/summary）
    :param rf: 无风险年化利率
    :return: dict 统计指标
    """
    metrics = {}
    # ---- 基于每日净值的指标 ----
    # 兼容rqalpha不同版本的pkl结构：4.x为total_portfolios，5.x+为portfolio
    df_port = result_dict.get('total_portfolios')
    if df_port is None:
        df_port = result_dict['portfolio']
    nv = df_port['unit_net_value'].astype(float)  # 每日单位净值
    daily_ret = nv.pct_change().dropna()  # 日收益率序列
    trading_days = len(daily_ret)
    total_returns = float(nv.iloc[-1] / nv.iloc[0] - 1)  # 累计收益率
    # 年化收益率（按252个交易日复合）
    annualized = float((1 + total_returns) ** (252 / trading_days) - 1) if trading_days > 0 else 0.0
    volatility = float(daily_ret.std() * np.sqrt(252))  # 年化波动率
    sharpe = (annualized - rf) / volatility if volatility > 0 else 0.0  # 夏普比率
    max_dd = float((nv / nv.cummax() - 1).min())  # 最大回撤（负值）
    calmar = annualized / abs(max_dd) if max_dd < 0 else 0.0  # 卡玛比率
    # ---- 基于交割单的交易统计 ----
    trade_returns = extract_trade_returns(result_dict)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns <= 0]
    win_rate = float(len(wins) / len(trade_returns)) if len(trade_returns) > 0 else 0.0  # 胜率
    pl_ratio = float(wins.mean() / abs(losses.mean())) if len(wins) > 0 and len(losses) > 0 else np.nan  # 盈亏比

    metrics['回测交易日数'] = trading_days
    metrics['累计收益率'] = total_returns
    metrics['年化收益率'] = annualized
    metrics['年化波动率'] = volatility
    metrics['夏普比率'] = sharpe
    metrics['最大回撤'] = max_dd
    metrics['卡玛比率'] = calmar
    metrics['平仓交易次数'] = len(trade_returns)
    metrics['胜率'] = win_rate
    metrics['盈亏比'] = pl_ratio
    metrics['平均盈利'] = float(wins.mean()) if len(wins) > 0 else np.nan
    metrics['平均亏损'] = float(losses.mean()) if len(losses) > 0 else np.nan
    return metrics


def print_analysis(metrics):
    """格式化输出回测结果分析指标"""
    rprint('\n========== 回测结果分析 ==========')
    rprint(f"回测交易日数 {metrics['回测交易日数']}"
           f"\n累计收益率 {metrics['累计收益率']:>.2%}\t年化收益率 {metrics['年化收益率']:>.2%}"
           f"\n年化波动率 {metrics['年化波动率']:>.2%}\t夏普比率 {metrics['夏普比率']:>.2f}"
           f"\n最大回撤 {metrics['最大回撤']:>.2%}\t卡玛比率 {metrics['卡玛比率']:>.2f}"
           f"\n平仓交易次数 {metrics['平仓交易次数']}\t胜率 {metrics['胜率']:>.2%}"
           f"\n盈亏比 {metrics['盈亏比']:>.2f}"
           f"\t平均盈利 {metrics['平均盈利']:>.2%}\t平均亏损 {metrics['平均亏损']:>.2%}")


start_time = f'程序开始时间：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'

if __name__ == '__main__':
    # 分析模式：python huice.py analyze [rq_result/xxx.pkl]
    # 不执行回测，直接分析已有的回测结果pkl文件，输出夏普比率、最大回撤、胜率等指标
    if 'analyze' in sys.argv[1:]:
        arg_index = sys.argv.index('analyze')
        pkl_path = sys.argv[arg_index + 1] if arg_index + 1 < len(sys.argv) else None
        if pkl_path is None:
            # 未指定文件时，使用rq_result目录下最新的pkl
            pkl_files = [os.path.join('rq_result', f) for f in os.listdir('rq_result') if f.endswith('.pkl')]
            pkl_path = max(pkl_files, key=os.path.getmtime)
        rprint(f'分析回测结果文件: {pkl_path}')
        print_analysis(analyze_result(pd.read_pickle(pkl_path)))
        sys.exit(0)

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
    rprint(
        f"回测起点 {result_dict['summary']['start_date']}"
        f"\n回测终点 {result_dict['summary']['end_date']}"
        f"\n回测收益 {result_dict['summary']['total_returns']:>.2%}\t年化收益 {result_dict['summary']['annualized_returns']:>.2%}"
        f"\t基准收益 {result_dict['summary']['benchmark_total_returns']:>.2%}\t基准年化 {result_dict['summary']['benchmark_annualized_returns']:>.2%}"
        f"\t最大回撤 {result_dict['summary']['max_drawdown']:>.2%}"
        f"\n打开程序文件夹下的rq_result.png查看收益走势图")
    # 回测结果分析：输出夏普比率、最大回撤、胜率等统计指标
    print_analysis(analyze_result(result_dict))
