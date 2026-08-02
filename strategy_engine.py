"""
多因子选股策略引擎
支持多个子策略的 AND / OR / NOT 逻辑组合，输出匹配的子策略列表和综合评分。
兼容现有的单策略模式（xuangu.py 默认行为不变）。

用法示例：
    from strategy_engine import StrategyEngine, StrategyNode, AND, OR, NOT

    # 定义策略节点（每个节点包装 CeLue.py 中的一个策略函数）
    node1 = StrategyNode("策略1快速", CeLue.策略1, weight=1.0, kwargs={"mode": "fast"})
    node2 = StrategyNode("策略2买入", CeLue.策略2, weight=2.0, extra_args=["HS300_信号"])

    # 逻辑组合
    combined = AND([node1, node2])
    engine = StrategyEngine(combined)

    # 对单只股票执行
    result = engine.evaluate(df_stock, HS300_信号=HS300_信号)
    # result.matched -> bool
    # result.matched_strategies -> ["策略1快速", "策略2买入"]
    # result.score -> float 综合评分
"""
import pandas as pd
import numpy as np


class StrategyResult:
    """策略评估结果容器"""

    def __init__(self, matched=False, matched_strategies=None, score=0.0, details=None):
        self.matched = matched
        self.matched_strategies = matched_strategies or []
        self.score = float(score)
        self.details = details or {}

    def __bool__(self):
        return self.matched

    def __repr__(self):
        return (f"StrategyResult(matched={self.matched}, "
                f"strategies={self.matched_strategies}, score={self.score:.3f})")


class StrategyNode:
    """
    单个策略节点，包装 CeLue.py 中的策略函数。

    :param name: 策略名称（用于结果展示）
    :param func: 策略函数，如 CeLue.策略1、CeLue.策略2
    :param weight: 权重，用于综合评分计算
    :param expected: True 表示函数返回 True 时视为命中；False 表示取反
    :param extra_args: 额外按位置传入的上下文参数名，例如 ["HS300_信号"]
                       这些参数会从 evaluate() 的关键字参数中查找并传入
    :param kwargs: 固定传入策略函数的关键字参数
    """

    def __init__(self, name, func, weight=1.0, expected=True, extra_args=None, **kwargs):
        self.name = name
        self.func = func
        self.weight = float(weight)
        self.expected = expected
        self.extra_args = extra_args or []
        self.static_kwargs = kwargs

    def evaluate(self, df, **context):
        # 构造调用参数
        call_args = []
        for arg_name in self.extra_args:
            if arg_name not in context:
                raise ValueError(f"策略 {self.name} 需要上下文参数 '{arg_name}'，但未提供")
            call_args.append(context[arg_name])

        call_kwargs = dict(self.static_kwargs)
        if 'start_date' in context and 'start_date' not in call_kwargs:
            call_kwargs['start_date'] = context.get('start_date', '')
        if 'end_date' in context and 'end_date' not in call_kwargs:
            call_kwargs['end_date'] = context.get('end_date', '')

        raw = self.func(df, *call_args, **call_kwargs)

        # 统一为布尔值：取最后一个元素（最新交易日）
        if isinstance(raw, pd.Series):
            if len(raw) == 0:
                hit = False
            else:
                val = raw.iat[-1]
                hit = bool(val) if not pd.isna(val) else False
        elif isinstance(raw, (bool, np.bool_)):
            hit = bool(raw)
        elif isinstance(raw, (np.integer, int)):
            hit = bool(raw)
        else:
            hit = bool(raw)

        if not self.expected:
            hit = not hit

        return StrategyResult(
            matched=hit,
            matched_strategies=[self.name] if hit else [],
            score=self.weight if hit else 0.0,
            details={self.name: hit},
        )


class _LogicNode:
    """逻辑组合节点基类"""

    def __init__(self, children, name=None, weight=1.0):
        self.children = children
        self.name = name or self.__class__.__name__
        self.weight = float(weight)

    def evaluate(self, df, **context):
        raise NotImplementedError


class AND(_LogicNode):
    """所有子策略都命中才命中"""

    def evaluate(self, df, **context):
        child_results = []
        all_matched = True
        total_score = 0.0
        details = {}
        for child in self.children:
            r = child.evaluate(df, **context)
            child_results.append(r)
            if not r.matched:
                all_matched = False
            total_score += r.score
            details.update(r.details)

        if all_matched:
            # 全部命中：收集所有子策略名称
            matched_strategies = []
            for r in child_results:
                matched_strategies.extend(r.matched_strategies)
            score = total_score
        else:
            # 未全部命中：只收集已命中子节点的策略名称（部分匹配信息）
            matched_strategies = []
            for r in child_results:
                if r.matched:
                    matched_strategies.extend(r.matched_strategies)
            score = total_score * 0.5  # 部分匹配给半权

        return StrategyResult(
            matched=all_matched,
            matched_strategies=matched_strategies,
            score=score,
            details=details,
        )


class OR(_LogicNode):
    """任意一个子策略命中即命中"""

    def evaluate(self, df, **context):
        any_matched = False
        matched_strategies = []
        total_score = 0.0
        details = {}
        for child in self.children:
            r = child.evaluate(df, **context)
            if r.matched:
                any_matched = True
                matched_strategies.extend(r.matched_strategies)
                total_score = max(total_score, r.score)
            details.update(r.details)
        return StrategyResult(
            matched=any_matched,
            matched_strategies=matched_strategies,
            score=total_score,
            details=details,
        )


class NOT(_LogicNode):
    """
    对单个子策略结果取反。
    :param weight: NOT 命中时（子策略未命中）的评分权重
    """

    def __init__(self, child, name=None, weight=1.0):
        super().__init__([child], name=name or "NOT", weight=weight)

    def evaluate(self, df, **context):
        child = self.children[0]
        r = child.evaluate(df, **context)
        matched = not r.matched
        if matched:
            # 子策略未命中，NOT 命中，给 NOT 自身权重分
            matched_strategies = [f"NOT({s})" for s in r.matched_strategies] if r.matched_strategies else [self.name]
            score = self.weight
        else:
            # 子策略命中，NOT 未命中
            matched_strategies = []
            score = 0.0
        return StrategyResult(
            matched=matched,
            matched_strategies=matched_strategies,
            score=score,
            details={f"NOT({k})": not v for k, v in r.details.items()},
        )


class StrategyEngine:
    """
    多因子策略引擎。

    :param root: 策略树的根节点（StrategyNode / AND / OR / NOT）
    """

    def __init__(self, root):
        self.root = root

    def evaluate(self, df, **context):
        """
        对单只股票执行策略树。

        :param df: 个股 DataFrame（需已设置日期索引）
        :param context: 上下文参数，如 HS300_信号、start_date、end_date
        :return: StrategyResult
        """
        return self.root.evaluate(df, **context)

    def evaluate_stocklist(self, stocklist, csvdaypath, df_today=None,
                           HS300_信号=None, df_gbbq=None,
                           start_date='', end_date='',
                           update_quote=False, tqdm_position=None):
        """
        批量评估股票列表，返回每个股票的结果字典。

        :return: dict {stockcode: StrategyResult}
        """
        import os
        import time
        from tqdm import tqdm
        import func

        results = {}
        tq = tqdm(stocklist, leave=False, position=tqdm_position) if tqdm_position is not None else tqdm(stocklist)
        for stockcode in tq:
            tq.set_description(stockcode)
            pklfile = csvdaypath + os.sep + stockcode + '.pkl'
            try:
                df_stock = pd.read_pickle(pklfile)
            except FileNotFoundError:
                continue

            if update_quote and df_today is not None:
                df_stock = func.update_stockquote(stockcode, df_stock, df_today)

            df_stock['date'] = pd.to_datetime(df_stock['date'], format='%Y-%m-%d')
            df_stock.set_index('date', drop=False, inplace=True)

            context = {
                'start_date': start_date,
                'end_date': end_date,
            }
            if HS300_信号 is not None:
                context['HS300_信号'] = HS300_信号
            if df_gbbq is not None:
                context['df_gbbq'] = df_gbbq

            result = self.evaluate(df_stock, **context)
            results[stockcode] = result

        return results


def build_default_engine():
    """
    构建与现有 xuangu.py 行为一致的默认引擎（策略1快速 + 策略2 AND 组合）。
    供单策略模式向后兼容使用。
    """
    import CeLue

    node_celue1 = StrategyNode(
        "策略1",
        CeLue.策略1,
        weight=1.0,
        mode='fast',
    )
    node_celue2 = StrategyNode(
        "策略2",
        CeLue.策略2,
        weight=2.0,
        extra_args=["HS300_信号"],
    )
    root = AND([node_celue1, node_celue2])
    return StrategyEngine(root)


def build_engine_from_config(config):
    """
    从字典配置构建策略引擎。

    配置格式示例：
    {
        "logic": "AND",
        "strategies": [
            {"name": "策略1", "func": "策略1", "weight": 1.0, "kwargs": {"mode": "fast"}},
            {"name": "策略2", "func": "策略2", "weight": 2.0, "extra_args": ["HS300_信号"]},
            {
                "logic": "OR",
                "strategies": [
                    {"name": "子策略A", "func": "..."},
                    {"name": "子策略B", "func": "..."},
                ]
            },
            {
                "logic": "NOT",
                "strategies": [
                    {"name": "排除策略", "func": "..."}
                ]
            }
        ]
    }

    :param config: dict
    :return: StrategyEngine
    """
    import CeLue

    def _build_node(node_cfg):
        if 'logic' in node_cfg:
            logic = node_cfg['logic'].upper()
            children = [_build_node(c) for c in node_cfg.get('strategies', [])]
            weight = node_cfg.get('weight', 1.0)
            node_name = node_cfg.get('name')
            if logic == 'AND':
                return AND(children, name=node_name, weight=weight)
            elif logic == 'OR':
                return OR(children, name=node_name, weight=weight)
            elif logic == 'NOT':
                if len(children) != 1:
                    raise ValueError("NOT 逻辑节点必须恰好包含一个子策略")
                return NOT(children[0], name=node_name, weight=weight)
            else:
                raise ValueError(f"不支持的逻辑类型: {logic}")
        else:
            func_name = node_cfg['func']
            if not hasattr(CeLue, func_name):
                raise ValueError(f"CeLue 模块中找不到策略函数: {func_name}")
            func = getattr(CeLue, func_name)
            return StrategyNode(
                name=node_cfg.get('name', func_name),
                func=func,
                weight=node_cfg.get('weight', 1.0),
                expected=node_cfg.get('expected', True),
                extra_args=node_cfg.get('extra_args', []),
                **node_cfg.get('kwargs', {}),
            )

    root = _build_node(config)
    return StrategyEngine(root)
