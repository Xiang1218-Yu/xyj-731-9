#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
命令行配置覆盖工具
==================

为 xuangu.py / monitor.py / celue_save.py / huice.py 提供统一的“命令行参数覆盖配置”能力，
使得 user_config.py 中的 strategy_combo、celue_db、monitor 等配置都可以在启动时用参数临时覆盖，
而无需修改配置文件。

支持的参数形式（均为 key=value，可与原有的位置开关参数如 single/del/combo 共存）：

    combo='{"op":"OR","children":["策略1","策略2"]}'   # JSON 覆盖 strategy_combo
    db=/path/to/xxx.db                                  # 覆盖策略信号数据库路径 celue_db
    interval=10                                         # 覆盖 monitor 轮询间隔
    watchlist=000001,600030                             # 覆盖 monitor 监控列表
    logfile=my_alert.log                                # 覆盖 monitor 预警日志文件

作者：功能迭代新增。
"""
import sys
import json

import user_config as ucfg


def parse_kv(argv=None):
    """把命令行里的 key=value 参数解析成 dict。忽略不含 '=' 的位置开关参数。"""
    if argv is None:
        argv = sys.argv[1:]
    kv = {}
    for arg in argv:
        if '=' in arg:
            key, value = arg.split('=', 1)
            kv[key.strip()] = value
    return kv


def get_strategy_combo(argv=None):
    """
    返回最终生效的组合策略配置。
    若命令行提供 combo=<JSON>，则解析 JSON 覆盖 user_config.strategy_combo。
    """
    kv = parse_kv(argv)
    if 'combo' in kv:
        try:
            return json.loads(kv['combo'])
        except (json.JSONDecodeError, ValueError) as e:
            print(f'[cli_config] combo 参数不是合法 JSON，忽略并使用默认配置: {e}')
    return ucfg.strategy_combo


def get_db_path(argv=None):
    """返回最终生效的策略信号数据库路径。命令行 db=<path> 优先。"""
    kv = parse_kv(argv)
    return kv.get('db', ucfg.celue_db)


def get_monitor_config(argv=None):
    """
    返回最终生效的监控配置 dict（interval/watchlist/log_file）。
    命令行 interval=、watchlist=、logfile= 优先覆盖 user_config.monitor。
    """
    kv = parse_kv(argv)
    cfg = dict(ucfg.monitor)  # 复制，避免修改全局配置
    if 'interval' in kv:
        try:
            cfg['interval'] = int(kv['interval'])
        except ValueError:
            print(f'[cli_config] interval 参数非整数，忽略: {kv["interval"]}')
    if 'watchlist' in kv:
        cfg['watchlist'] = [c for c in kv['watchlist'].split(',') if c]
    if 'logfile' in kv:
        cfg['log_file'] = kv['logfile']
    return cfg
