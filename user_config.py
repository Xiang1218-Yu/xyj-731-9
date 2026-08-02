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

# 多因子选股组合配置（供 xuangu.py 的多因子模式 与 monitor.py 使用）
# 语法见 CeLue_engine.build_node：字符串=单因子；{'op':'AND/OR/NOT', ...}=逻辑组合。
# 默认等价于原“策略1 AND 策略2”串联逻辑，保证向后兼容。
strategy_combo = {
    'op': 'AND',
    'children': ['策略1', '策略2'],
}

# 实时监控模块 monitor.py 配置
monitor = {
    'interval': 30,                 # 行情轮询间隔（秒）
    'watchlist': [],                # 监控股票列表，空则监控 csv_lday 目录全部股票
    'log_file': 'monitor_alert.log',  # 预警日志文件
}

# 策略信号 SQLite 数据库路径（celue_save.py 写入，huice.py 读取，形成闭环）
celue_db = tdx['csv_gbbq'] + '/celue汇总.db'

# 配置部分结束
