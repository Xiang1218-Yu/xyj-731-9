#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
多因子选股策略引擎（strategy_engine）

设计目标：
1. 兼容现有 CeLue.py 中的单策略模式（策略1 / 策略2 / 卖策略 / 策略HS300）。
2. 支持同时配置多个子策略（因子），并通过 AND / OR / NOT 三种逻辑进行任意嵌套组合。
3. 每个子策略可设置权重，组合结果输出：
   - matched:      最终是否匹配（布尔值）
   - matched_strategies: 命中的子策略名称列表
   - score:        命中子策略的权重之和（原始评分）
   - score_normalized: 原始评分 / 全部叶子权重之和（0~1 的综合评分）
4. 同时支持“单日选股”（返回标量 bool）和“回测序列计算”（返回 pandas.Series）两种场景。

使用方式示例：
    from strategy_engine import Strategy, CompositeStrategy, StrategyEngine

    s1 = Strategy('快速筛选', CeLue.策略1, weight=1.0, strategy_kwargs={'mode': 'fast'})
    s2 = Strategy('详细筛选', CeLue.策略2, weight=2.0, needs_hs300=True)
    s3 = Strategy('涨停排除', CeLue.策略1, weight=0.5, strategy_kwargs={'mode': 'fast'})

    root = CompositeStrategy('组合策略', op='AND', children=[
        s1,
        CompositeStrategy('或', op='OR', children=[s2]),
        CompositeStrategy('非', op='NOT', children=[s3]),
    ])

    engine = StrategyEngine(root)
    result = engine.evaluate_stock(df_stock, hs300_signal=HS300_信号)

也可以直接通过 dict 配置（便于 JSON / 启动参数传入）：
    cfg = {"name": "组合策略", "op": "AND", "children": [
        {"name": "快速筛选", "func": "策略1", "kwargs": {"mode": "fast"}, "weight": 1.0},
        {"name": "详细筛选", "func": "策略2", "needs_hs300": true, "weight": 2.0},
    ]}
    engine = StrategyEngine.from_dict(cfg)
"""

import pandas as pd
import numpy as np


class StrategyResult:
    """
    策略节点求值结果。

    signal:  bool 或 pandas.Series(bool)。本节点是否命中。
    weight_sum: 若命中，命中的叶子节点权重之和；未命中为 0。NOT 节点不计入权重。
    matched_names: 命中的叶子节点名称列表（去重，保持插入顺序）。
    total_weight: 本子树全部叶子节点的权重之和（用于归一化）。
    """

    __slots__ = ('signal', 'weight_sum', 'matched_names', 'total_weight')

    def __init__(self, signal, weight_sum=0.0, matched_names=None, total_weight=0.0):
        self.signal = signal
        self.weight_sum = weight_sum
        self.matched_names = matched_names if matched_names is not None else []
        self.total_weight = total_weight


class BaseStrategy:
    """策略节点基类，组合模式。"""

    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = float(weight)

    def evaluate(self, df, context=None):
        """
        对单只股票的行情数据求值。
        :param df: 单只股票 DataFrame，date 列为索引。
        :param context: dict，存放跨策略共享数据，如 {'hs300_signal': Series}。
        :return: StrategyResult
        """
        raise NotImplementedError

    def total_leaf_weight(self):
        """本子树全部叶子节点权重之和。"""
        raise NotImplementedError


class Strategy(BaseStrategy):
    """
    叶子策略节点：包装 CeLue.py 中的一个策略函数。

    :param name:             策略名称，用于结果输出。
    :param func:             可调用对象，签名形如 func(df, ...) -> bool/Series。
                             若传入字符串，则延迟到第一次求值时从 CeLue 模块中按名称解析。
    :param weight:           权重，用于综合评分。
    :param strategy_args:    传递给策略函数的位置参数。
    :param strategy_kwargs:  传递给策略函数的关键字参数。
    :param needs_hs300:      是否需要把 HS300 信号作为第一个位置参数传入（适配 策略2）。
    :param is_sell:          是否为卖出策略（由监控/回测模块使用，本引擎不做特殊处理）。
    """

    def __init__(self, name, func, weight=1.0, strategy_args=None, strategy_kwargs=None,
                 needs_hs300=False, is_sell=False):
        super().__init__(name, weight)
        self._func = func
        self._func_resolved = callable(func)
        self.strategy_args = strategy_args if strategy_args is not None else []
        self.strategy_kwargs = strategy_kwargs if strategy_kwargs is not None else {}
        self.needs_hs300 = needs_hs300
        self.is_sell = is_sell

    def _resolve_func(self):
        """延迟导入并解析 CeLue 模块中的策略函数，避免循环导入。"""
        if self._func_resolved:
            return self._func
        import CeLue  # 个人策略文件，延迟导入
        if not hasattr(CeLue, self._func):
            raise AttributeError(f'CeLue 模块中不存在名为 "{self._func}" 的策略函数')
        self._func = getattr(CeLue, self._func)
        self._func_resolved = True
        return self._func

    def evaluate(self, df, context=None):
        if context is None:
            context = {}
        func = self._resolve_func()
        args = list(self.strategy_args)
        # 需要 HS300 信号的策略（如 策略2），把信号作为第一个位置参数传入
        if self.needs_hs300:
            hs300_signal = context.get('hs300_signal')
            if hs300_signal is None:
                raise ValueError(f'策略 "{self.name}" 需要 HS300 信号，但 context 中未提供 hs300_signal')
            args.insert(0, hs300_signal)
        signal = func(df, *args, **self.strategy_kwargs)
        # 统一为布尔类型
        if isinstance(signal, pd.Series):
            signal = signal.astype(bool).fillna(False)
        else:
            signal = bool(signal)
        return StrategyResult(
            signal=signal,
            weight_sum=self.weight,
            matched_names=[self.name],
            total_weight=self.weight,
        )

    def total_leaf_weight(self):
        return self.weight


class CompositeStrategy(BaseStrategy):
    """
    组合策略节点：对多个子节点的结果做 AND / OR / NOT 逻辑运算。

    - AND: 全部子节点命中才命中；命中时权重 = 所有命中子节点的权重之和。
    - OR : 任意子节点命中即命中；权重 = 所有命中子节点的权重之和。
    - NOT: 只允许一个子节点；结果取反；不计入权重（作为过滤器使用）。
    """

    def __init__(self, name, op, children=None, weight=1.0):
        super().__init__(name, weight)
        op = op.upper()
        if op not in ('AND', 'OR', 'NOT'):
            raise ValueError(f'不支持的逻辑运算符: {op}，仅支持 AND / OR / NOT')
        if op == 'NOT' and children and len(children) != 1:
            raise ValueError('NOT 逻辑节点只允许包含 1 个子节点')
        self.op = op
        self.children = children if children is not None else []

    def add(self, child):
        if self.op == 'NOT' and len(self.children) >= 1:
            raise ValueError('NOT 逻辑节点只允许包含 1 个子节点')
        self.children.append(child)
        return self

    def total_leaf_weight(self):
        # NOT 作为过滤器，其权重不计入综合评分的分母
        if self.op == 'NOT':
            return 0.0
        return sum(c.total_leaf_weight() for c in self.children)

    @staticmethod
    def _combine_names(results, signal_mask=None):
        """合并命中的策略名称，去重并保持顺序。"""
        seen = []
        for r in results:
            # Series 场景下，只要本序列存在任意 True，即把该叶子名称计入（用于整体选股列表展示）
            include = False
            if signal_mask is None:
                include = True
            elif isinstance(signal_mask, bool):
                include = signal_mask
            else:
                include = bool(signal_mask.any()) if hasattr(signal_mask, 'any') else bool(signal_mask)
            if include:
                for n in r.matched_names:
                    if n not in seen:
                        seen.append(n)
        return seen

    def evaluate(self, df, context=None):
        if not self.children:
            return StrategyResult(signal=False, weight_sum=0.0, total_weight=self.total_leaf_weight())

        results = [c.evaluate(df, context) for c in self.children]
        signals = [r.signal for r in results]

        if self.op == 'NOT':
            child = results[0]
            if isinstance(child.signal, pd.Series):
                signal = ~child.signal.astype(bool)
            else:
                signal = not bool(child.signal)
            # NOT 作为过滤器，不计入权重
            return StrategyResult(
                signal=signal,
                weight_sum=0.0,
                matched_names=[],
                total_weight=self.total_leaf_weight(),
            )

        if self.op == 'AND':
            if isinstance(signals[0], pd.Series):
                signal = signals[0]
                for s in signals[1:]:
                    signal = signal & s.astype(bool)
            else:
                signal = all(bool(s) for s in signals)
        else:  # OR
            if isinstance(signals[0], pd.Series):
                signal = signals[0]
                for s in signals[1:]:
                    signal = signal | s.astype(bool)
            else:
                signal = any(bool(s) for s in signals)

        # 计算命中权重和
        if isinstance(signal, pd.Series):
            weight_sum = pd.Series(0.0, index=signal.index)
            for r in results:
                if isinstance(r.signal, pd.Series):
                    weight_sum = weight_sum + r.signal.astype(float) * r.weight_sum
                elif bool(r.signal):
                    weight_sum = weight_sum + r.weight_sum
            matched_names = self._combine_names(results, signal)
        else:
            weight_sum = sum(r.weight_sum for r in results if bool(r.signal))
            matched_names = self._combine_names(
                results,
                [bool(r.signal) for r in results],
            )

        return StrategyResult(
            signal=signal,
            weight_sum=weight_sum,
            matched_names=matched_names,
            total_weight=self.total_leaf_weight(),
        )


class StrategyEngine:
    """
    策略引擎：对一组股票批量执行根策略，并整理输出。

    兼容旧版单策略模式：若不传 root_strategy，则 evaluate_stock 会直接调用 策略1(mode=fast)，
    与重构前 xuangu.run_celue1 的行为完全一致。
    """

    def __init__(self, root_strategy=None):
        self.root = root_strategy

    # ------------------------------------------------------------------ #
    # 工厂方法：从 dict 配置构建引擎
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, config):
        """
        从嵌套 dict 构建策略树。
        叶子节点字段：name / func / weight / args / kwargs / needs_hs300 / is_sell
        组合节点字段：name / op / children
        """
        if config is None:
            return cls(None)

        def _build(node):
            if 'op' in node and node['op'] is not None:
                comp = CompositeStrategy(
                    name=node.get('name', node['op']),
                    op=node['op'],
                    weight=node.get('weight', 1.0),
                )
                for child_cfg in node.get('children', []):
                    comp.add(_build(child_cfg))
                return comp
            else:
                return Strategy(
                    name=node['name'],
                    func=node.get('func', node['name']),
                    weight=node.get('weight', 1.0),
                    strategy_args=node.get('args', []),
                    strategy_kwargs=node.get('kwargs', {}),
                    needs_hs300=node.get('needs_hs300', False),
                    is_sell=node.get('is_sell', False),
                )

        return cls(_build(config))

    @classmethod
    def default_single_strategy(cls):
        """
        构建与旧版 xuangu.py 完全一致的默认策略：
        先跑 策略1(mode=fast)，再跑 策略2（依赖 HS300 信号），两者 AND。
        """
        root = CompositeStrategy('默认单策略组合', op='AND', children=[
            Strategy('策略1快速筛选', '策略1', weight=1.0, strategy_kwargs={'mode': 'fast'}),
            Strategy('策略2详细筛选', '策略2', weight=1.0, needs_hs300=True),
        ])
        return cls(root)

    # ------------------------------------------------------------------ #
    # 核心求值
    # ------------------------------------------------------------------ #
    def evaluate_stock(self, df_stock, hs300_signal=None, start_date='', end_date=''):
        """
        对单只股票求值（单日选股场景）。

        :return: dict，字段：
                 stock, matched, matched_strategies, score, score_normalized
        """
        context = {'hs300_signal': hs300_signal}

        if start_date or end_date:
            sd = start_date if start_date else df_stock.index[0]
            ed = end_date if end_date else df_stock.index[-1]
            df_eval = df_stock.loc[sd:ed]
        else:
            df_eval = df_stock

        # 兼容模式：没有配置多因子时，直接使用旧版 策略1(fast) 行为
        if self.root is None:
            import CeLue
            matched = CeLue.策略1(df_eval, start_date=start_date, end_date=end_date, mode='fast')
            matched = bool(matched)
            return {
                'stock': df_eval['code'].iloc[0] if 'code' in df_eval.columns else '',
                'matched': matched,
                'matched_strategies': ['策略1'] if matched else [],
                'score': 1.0 if matched else 0.0,
                'score_normalized': 1.0 if matched else 0.0,
            }

        result = self.root.evaluate(df_eval, context)
        signal = result.signal
        # 单日场景取最后一个值
        if isinstance(signal, pd.Series):
            matched = bool(signal.iat[-1])
            score = float(result.weight_sum.iat[-1]) if isinstance(result.weight_sum, pd.Series) else float(result.weight_sum)
        else:
            matched = bool(signal)
            score = float(result.weight_sum)

        total_weight = self.root.total_leaf_weight()
        score_norm = round(score / total_weight, 4) if total_weight > 0 else 0.0

        return {
            'stock': df_eval['code'].iloc[0] if 'code' in df_eval.columns else '',
            'matched': matched,
            'matched_strategies': list(result.matched_names) if matched else [],
            'score': round(score, 4),
            'score_normalized': score_norm,
        }

    def evaluate_stock_series(self, df_stock, hs300_signal=None, start_date='', end_date=''):
        """
        对单只股票求值，返回完整的信号 Series（回测场景使用）。

        :return: dict，字段：
                 code, buy_signal(Series), matched_strategies(list),
                 score(Series), score_normalized(Series)
        """
        context = {'hs300_signal': hs300_signal}

        if start_date or end_date:
            sd = start_date if start_date else df_stock.index[0]
            ed = end_date if end_date else df_stock.index[-1]
            df_eval = df_stock.loc[sd:ed]
        else:
            df_eval = df_stock

        if self.root is None:
            import CeLue
            signal = CeLue.策略2(df_eval, hs300_signal, start_date=start_date, end_date=end_date)
            return {
                'code': df_eval['code'].iloc[0] if 'code' in df_eval.columns else '',
                'buy_signal': signal.astype(bool),
                'matched_strategies': ['策略2'],
                'score': signal.astype(float),
                'score_normalized': signal.astype(float),
            }

        result = self.root.evaluate(df_eval, context)
        signal = result.signal
        if not isinstance(signal, pd.Series):
            signal = pd.Series(signal, index=df_eval.index)
        score = result.weight_sum
        if not isinstance(score, pd.Series):
            score = pd.Series(score, index=df_eval.index)
        total_weight = self.root.total_leaf_weight()
        if total_weight > 0:
            score_norm = (score / total_weight).round(4)
        else:
            score_norm = pd.Series(0.0, index=df_eval.index)

        return {
            'code': df_eval['code'].iloc[0] if 'code' in df_eval.columns else '',
            'buy_signal': signal.astype(bool),
            'matched_strategies': list(result.matched_names),
            'score': score,
            'score_normalized': score_norm,
        }

    def get_sell_strategy_names(self):
        """返回树中被标记为 is_sell 的叶子策略名称列表（供监控/回测识别卖出策略）。"""
        names = []

        def _walk(node):
            if isinstance(node, Strategy):
                if node.is_sell and node.name not in names:
                    names.append(node.name)
            elif isinstance(node, CompositeStrategy):
                for c in node.children:
                    _walk(c)

        if self.root is not None:
            _walk(self.root)
        return names


# ---------------------------------------------------------------------- #
# 便捷的 JSON 配置加载工具
# ---------------------------------------------------------------------- #
def load_engine_from_json(json_path):
    """从 JSON 文件加载策略树配置，返回 StrategyEngine。"""
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return StrategyEngine.from_dict(config)
