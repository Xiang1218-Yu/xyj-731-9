#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
多因子选股策略引擎
================

本模块在不修改原有 CeLue.py 策略函数的前提下，对其进行封装，
提供“多因子组合选股”能力：

1. 允许同时配置多个子策略（因子），并使用 AND / OR / NOT 逻辑任意组合；
2. 完全兼容原有的单策略模式（单个因子即等价于原来的单策略调用）；
3. 组合结果输出：每只股票命中的子策略列表 + 综合评分。

设计要点
--------
* 每个子策略被封装为一个“因子适配器”，统一把 CeLue.py 中形态各异的返回值
  （有的返回 bool，有的返回 pd.Series）归一化为“布尔信号序列 pd.Series”。
* 组合逻辑用一棵表达式树描述（AndNode / OrNode / NotNode / FactorNode），
  既能表达简单的单因子，也能表达任意嵌套的 AND/OR/NOT 组合。
* 评分（score）采用“命中因子的加权占比”：默认每个因子权重为 1，
  综合评分 = 命中因子权重之和 / 参与评分的因子权重之和，取值 0~1。

作者：功能迭代新增，风格对齐项目现有中文命名习惯。
"""

import pandas as pd

import CeLue  # 复用原有个人策略文件，不改动其内容


# ---------------------------------------------------------------------------
# 因子适配层：把 CeLue.py 里返回值不统一的策略函数，统一为“布尔信号序列”
# ---------------------------------------------------------------------------
def _to_bool_series(result, index):
    """
    把子策略的返回值归一化为与 df 对齐的布尔序列 pd.Series。

    CeLue.py 中：
      - 策略1(mode='fast') 返回单个 bool；
      - 策略1(普通模式)/策略2/策略HS300 返回 pd.Series。
    统一处理，保证组合引擎可以对任意因子做逻辑运算。
    """
    if isinstance(result, pd.Series):
        # 已是序列，转为布尔并填充空值为 False
        return result.reindex(index).fillna(False).astype(bool)
    # 标量 bool：广播到整个索引（末位代表“当日是否命中”）
    series = pd.Series(False, index=index, dtype=bool)
    if bool(result):
        series.iloc[-1] = True
    return series


def _factor_celue1(df, context):
    """因子适配器：策略1（沿用盘中选股的 fast 模式，保持与 xuangu.py 一致）"""
    result = CeLue.策略1(df, start_date=context.get('start_date', ''),
                         end_date=context.get('end_date', ''), mode='fast')
    return _to_bool_series(result, df.index)


def _factor_celue1_full(df, context):
    """因子适配器：策略1 完整模式（返回序列，适合回测/信号保存场景）"""
    result = CeLue.策略1(df, start_date=context.get('start_date', ''),
                         end_date=context.get('end_date', ''))
    return _to_bool_series(result, df.index)


def _factor_celue2(df, context):
    """因子适配器：策略2（需要 HS300_信号，通过 context 传入）"""
    result = CeLue.策略2(df, context.get('HS300_信号'),
                         start_date=context.get('start_date', ''),
                         end_date=context.get('end_date', ''))
    return _to_bool_series(result, df.index)


# 子策略注册表：名称 -> 适配器。新增策略时在此登记即可被组合引擎调用。
STRATEGY_REGISTRY = {
    '策略1': _factor_celue1,
    '策略1_full': _factor_celue1_full,
    '策略2': _factor_celue2,
}


def register_strategy(name, adapter):
    """对外注册自定义子策略适配器。adapter 签名为 adapter(df, context) -> pd.Series[bool]"""
    STRATEGY_REGISTRY[name] = adapter


# ---------------------------------------------------------------------------
# 组合表达式树：FactorNode(叶子) / AndNode / OrNode / NotNode
# ---------------------------------------------------------------------------
class Node:
    """表达式树节点基类。evaluate 返回 (布尔信号序列, {因子名: 布尔信号序列})"""

    def evaluate(self, df, context):
        raise NotImplementedError


class FactorNode(Node):
    """叶子节点：单个子策略因子"""

    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight  # 该因子在综合评分中的权重

    def evaluate(self, df, context):
        if self.name not in STRATEGY_REGISTRY:
            raise KeyError(f'未注册的子策略: {self.name}，可用: {list(STRATEGY_REGISTRY)}')
        signal = STRATEGY_REGISTRY[self.name](df, context)
        return signal, {self.name: signal}


class AndNode(Node):
    """AND：所有子节点信号按位与"""

    def __init__(self, children):
        self.children = children

    def evaluate(self, df, context):
        result = None
        factors = {}
        for child in self.children:
            sig, sub = child.evaluate(df, context)
            factors.update(sub)
            result = sig if result is None else (result & sig)
        return result, factors


class OrNode(Node):
    """OR：所有子节点信号按位或"""

    def __init__(self, children):
        self.children = children

    def evaluate(self, df, context):
        result = None
        factors = {}
        for child in self.children:
            sig, sub = child.evaluate(df, context)
            factors.update(sub)
            result = sig if result is None else (result | sig)
        return result, factors


class NotNode(Node):
    """NOT：对唯一子节点信号取反"""

    def __init__(self, child):
        self.child = child

    def evaluate(self, df, context):
        sig, factors = self.child.evaluate(df, context)
        # 同步对子节点的因子信号取反：NOT 之后，因子的“有效贡献”应为其取反结果。
        # 否则被 NOT 排除的因子仍会被记为命中，导致 matched_factors 和 score 与
        # 组合结果不一致（例如 NOT 命中时 matched=False 却 score=1.0）。
        negated_factors = {name: ~s for name, s in factors.items()}
        return ~sig, negated_factors


# ---------------------------------------------------------------------------
# 配置解析：把 dict 配置转成表达式树
# ---------------------------------------------------------------------------
def build_node(config):
    """
    根据配置构建表达式树。支持两种写法，可任意嵌套：

    1) 单策略（兼容原单策略模式），字符串或带权重的 dict：
         '策略1'
         {'factor': '策略1', 'weight': 2.0}

    2) 逻辑组合，dict 形式：
         {'op': 'AND', 'children': [<config>, <config>, ...]}
         {'op': 'OR',  'children': [...]}
         {'op': 'NOT', 'child': <config>}

    示例（策略1 AND 策略2 AND (NOT 策略1_full)）：
         {'op': 'AND', 'children': [
             '策略1', '策略2',
             {'op': 'NOT', 'child': '策略1_full'},
         ]}
    """
    # 字符串：等价于单因子叶子
    if isinstance(config, str):
        return FactorNode(config)

    if isinstance(config, dict):
        # 叶子写法：{'factor': 名称, 'weight': 权重}
        if 'factor' in config:
            return FactorNode(config['factor'], weight=config.get('weight', 1.0))

        # 逻辑节点写法
        op = str(config.get('op', '')).upper()
        if op == 'AND':
            return AndNode([build_node(c) for c in config['children']])
        if op == 'OR':
            return OrNode([build_node(c) for c in config['children']])
        if op == 'NOT':
            return NotNode(build_node(config['child']))
        raise ValueError(f'不支持的逻辑操作符: {config.get("op")}，仅支持 AND/OR/NOT')

    raise TypeError(f'无法解析的策略配置类型: {type(config)}')


# ---------------------------------------------------------------------------
# 引擎入口
# ---------------------------------------------------------------------------
class StrategyEngine:
    """
    多因子策略引擎。

    用法::

        engine = StrategyEngine(config)          # config 见 build_node 说明
        result = engine.run(df_stock, context)   # context 里放 HS300_信号 等

    其中 config 既可以是单策略（兼容旧模式），也可以是 AND/OR/NOT 组合。
    """

    def __init__(self, config):
        self.config = config
        self.root = build_node(config)
        # 收集所有参与评分的叶子因子及其权重
        self._leaf_weights = {}
        self._collect_leaves(self.root)

    def _collect_leaves(self, node):
        if isinstance(node, FactorNode):
            self._leaf_weights[node.name] = node.weight
        elif isinstance(node, (AndNode, OrNode)):
            for c in node.children:
                self._collect_leaves(c)
        elif isinstance(node, NotNode):
            self._collect_leaves(node.child)

    def run(self, df, context=None):
        """
        对单只股票 DataFrame 执行组合策略。

        返回 dict：
          {
            'matched': bool,             # 组合逻辑当日是否命中
            'matched_factors': [名称],   # 当日命中的子策略列表
            'score': float,              # 综合评分 0~1
          }
        """
        context = context or {}
        combined, factors = self.root.evaluate(df, context)

        # 组合逻辑当日结果（最后一个交易日）
        matched = bool(combined.iloc[-1]) if len(combined) else False

        # 当日命中的子策略列表
        matched_factors = [name for name, sig in factors.items()
                           if len(sig) and bool(sig.iloc[-1])]

        # 综合评分 = 命中因子权重和 / 全部因子权重和
        total_weight = sum(self._leaf_weights.values()) or 1.0
        hit_weight = sum(self._leaf_weights.get(name, 1.0) for name in matched_factors)
        score = round(hit_weight / total_weight, 4)

        return {
            'matched': matched,
            'matched_factors': matched_factors,
            'score': score,
        }


# 默认组合配置：等价于原 xuangu.py 的“策略1 AND 策略2”串联逻辑，
# 作为向后兼容的默认值使用。
DEFAULT_CONFIG = {
    'op': 'AND',
    'children': ['策略1', '策略2'],
}
