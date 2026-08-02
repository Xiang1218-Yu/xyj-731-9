#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
用户设置文件

作者：wking [http://wkings.net]
"""

# 配置部分开始
debug = False  # 是否开启调试日志输出  开=True  关=False

# 目录最好事先手动建立好，不然程序会出错
# 修改为当前项目目录下的相对路径，适配macOS
tdx = {
    'tdx_path': '/Users/tog/Desktop/code/gsb/gsb-731/xyj-731-9/xyj-731-9/data/tdx',  # 指定通达信目录
    'csv_lday': '/Users/tog/Desktop/code/gsb/gsb-731/xyj-731-9/xyj-731-9/data/lday_qfq',  # 指定csv格式日线数据保存目录
    'pickle': '/Users/tog/Desktop/code/gsb/gsb-731/xyj-731-9/xyj-731-9/data/pickle',  # 指定pickle格式日线数据保存目录
    'csv_index': '/Users/tog/Desktop/code/gsb/gsb-731/xyj-731-9/xyj-731-9/data/index',  # 指定指数保存目录
    'csv_cw': '/Users/tog/Desktop/code/gsb/gsb-731/xyj-731-9/xyj-731-9/data/cw',  # 指定专业财务保存目录
    'csv_gbbq': '/Users/tog/Desktop/code/gsb/gsb-731/xyj-731-9/xyj-731-9/data',  # 指定股本变迁保存目录
    'pytdx_ip': '218.6.170.55',  # 指定pytdx的通达信服务器IP
    'pytdx_port': 7709,  # 指定pytdx的通达信服务器端口。int类型
}

index_list = [  # 通达信需要转换的指数文件。通达信按998查看重要指数
    'sh999999.day',  # 上证指数
    'sh000300.day',  # 沪深300
    'sz399001.day',  # 深成指
]

# ==================== 多因子选股配置（celue_engine.py 读取） ====================
# 多因子配置接口。xuangu.py 带 multi 参数、monitor.py 带 --multi 参数时生效
# factors 说明：
#   name    策略名，需在 celue_engine.STRATEGY_REGISTRY 中已注册
#   weight  因子权重，用于计算综合评分（命中因子权重和/总权重）
#   not     True表示对该因子取反（NOT逻辑），例如卖策略出现信号则否定该股
#   enabled 是否启用该因子
# logic 说明：
#   'AND' 全部因子命中才入选；'OR' 任一因子命中即入选；
#   也可写表达式，如 "策略1 AND (策略2 OR NOT 卖策略)"，AND/OR/NOT不区分大小写
# score_threshold 综合评分门槛(0~1)，逻辑组合通过后评分还需达到该值才入选，0表示只看逻辑结果
multi_factor = {
    'factors': [
        {'name': '策略1', 'weight': 1.0, 'not': False, 'enabled': True},
        {'name': '策略2', 'weight': 2.0, 'not': False, 'enabled': True},
        {'name': '卖策略', 'weight': 1.0, 'not': True, 'enabled': False},  # NOT用法示例，默认停用
        {'name': 'MA多头', 'weight': 1.0, 'not': False, 'enabled': False},  # 内置示例因子，默认停用
        {'name': '放量突破', 'weight': 1.0, 'not': False, 'enabled': False},  # 内置示例因子，默认停用
    ],
    'logic': 'AND',
    'score_threshold': 0.0,
}

# ==================== 实时监控配置（monitor.py 读取） ====================
monitor = {
    'watchlist': [],  # 监控股票列表，如 ['000001', '600030']。为空则使用celue汇总中选出的股票
    'interval': 30,  # 行情获取间隔秒数，默认30秒
    'use_multi_factor': False,  # True=监控时使用多因子引擎判定买入信号，False=原策略2单策略
    'alert_csv': 'monitor_signals.csv',  # 预警信号记录文件（信号时间、代码、价格、触发策略）
    'alert_log': 'monitor_alert.log',  # 预警日志文件
}

# 配置部分结束
