"""
多因子选股策略引擎

在现有单策略（CeLue.py 策略1/策略2/卖策略）基础上重构的策略执行模块：
1. 支持同时配置多个策略（因子），每个因子可配置权重与取反(NOT)标志
2. 支持 AND / OR / NOT 逻辑组合：
   - logic = 'AND'  全部因子命中才入选
   - logic = 'OR'   任一因子命中即入选
   - logic = 表达式 例如 "策略1 AND (策略2 OR NOT 卖策略)"
3. 兼容现有单策略模式：配置中只启用一个因子（logic='AND'）时等同于原单策略；
   也可使用 run_single_strategy() 直接按原方式调用单个策略
4. 输出每只股票匹配的子策略列表和综合评分（命中因子权重和 / 总权重，0~1）

因子适配器统一签名：func(df, context) -> pd.Series(bool)，索引与df一致
context 为 dict，常用键：
    'HS300_信号'  HS300大盘信号序列（策略2依赖）
    'start_date' / 'end_date'  策略计算区间
    '_cache'      因子序列缓存（避免卖策略重复计算策略2）
"""
import re
import numpy as np
import pandas as pd
from rich import print

import CeLue  # 个人策略文件，不分享
from func_TDX import SMA
import user_config as ucfg


# ======================== 因子适配器（包装CeLue.py现有策略） ========================

def _factor_celue1(df, context):
    """策略1因子 - 快速筛选条件的序列模式（非fast模式，返回布尔序列）"""
    return CeLue.策略1(df,
                       start_date=context.get('start_date', ''),
                       end_date=context.get('end_date', ''))


def _factor_celue2(df, context):
    """策略2因子 - 详细筛选（依赖HS300大盘信号，返回布尔序列）"""
    celue2 = CeLue.策略2(df, context.get('HS300_信号'),
                         start_date=context.get('start_date', ''),
                         end_date=context.get('end_date', ''))
    # 缓存策略2序列，卖策略因子直接复用，避免重复计算
    context.setdefault('_cache', {})['策略2'] = celue2
    return celue2


def _factor_sell(df, context):
    """卖策略因子 - 卖出信号序列。一般配合 not=True 使用（出现卖出信号则否定该股）"""
    cache = context.setdefault('_cache', {})
    if '策略2' not in cache:
        cache['策略2'] = CeLue.策略2(df, context.get('HS300_信号'),
                                     start_date=context.get('start_date', ''),
                                     end_date=context.get('end_date', ''))
    return CeLue.卖策略(df, cache['策略2'],
                        start_date=context.get('start_date', ''),
                        end_date=context.get('end_date', ''))


# ======================== 内置示例因子（可在user_config中按名启用/停用） ========================

def _factor_ma_long(df, context):
    """内置示例因子：MA多头排列（MA5>MA10>MA20）"""
    C = df['close']
    return (SMA(C, 5) > SMA(C, 10)) & (SMA(C, 10) > SMA(C, 20))


def _factor_vol_break(df, context):
    """内置示例因子：放量收阳（成交量大于前5日均量2倍且收盘价高于开盘价）"""
    return (df['vol'] > df['vol'].rolling(5).mean().shift(1) * 2) & (df['close'] > df['open'])


# 策略注册表：因子名 -> (适配函数, 说明)。自定义策略在此注册后即可在配置中使用
STRATEGY_REGISTRY = {
    '策略1': (_factor_celue1, 'CeLue.策略1 基础筛选'),
    '策略2': (_factor_celue2, 'CeLue.策略2 买入信号'),
    '卖策略': (_factor_sell, 'CeLue.卖策略 卖出信号(建议NOT使用)'),
    'MA多头': (_factor_ma_long, '内置因子 MA5>MA10>MA20'),
    '放量突破': (_factor_vol_break, '内置因子 放量收阳'),
}


def run_single_strategy(name, df, context=None):
    """
    兼容现有单策略模式的调用接口
    :param name: 策略注册表中的策略名，如 '策略1'、'策略2'、'卖策略'
    :param df: 个股日线DF（时间为索引）
    :param context: 可选，策略上下文字典
    :return: 布尔序列
    """
    if context is None:
        context = {}
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f'未注册的策略: {name}，可用策略: {list(STRATEGY_REGISTRY.keys())}')
    func, _ = STRATEGY_REGISTRY[name]
    return func(df, context)


# ======================== AND/OR/NOT 逻辑表达式解析器 ========================

class _LogicParser:
    """
    递归下降解析逻辑表达式。语法：
        expr    := or_expr
        or_expr := and_expr (OR and_expr)*
        and_expr:= unary (AND unary)*
        unary   := NOT unary | primary
        primary := 因子名 | '(' or_expr ')'
    关键字 AND/OR/NOT 不区分大小写，因子名支持中文
    """

    _token_re = re.compile(r'\s*(AND|OR|NOT|\(|\)|[^()\s]+)', re.IGNORECASE)

    def __init__(self, expression):
        self.tokens = [m.group(1) for m in self._token_re.finditer(expression)]
        self.pos = 0

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self):
        tok = self._peek()
        self.pos += 1
        return tok

    def parse(self, values):
        """values: dict 因子名->bool。返回表达式求值结果。解析器可复用，每次从头解析"""
        self.pos = 0
        if not self.tokens:
            raise ValueError('逻辑表达式为空')
        result = self._or_expr(values)
        if self._peek() is not None:
            raise ValueError(f'逻辑表达式存在多余内容: {self._peek()}')
        return result

    def _or_expr(self, values):
        result = self._and_expr(values)
        while self._peek() is not None and self._peek().upper() == 'OR':
            self._next()
            rhs = self._and_expr(values)
            result = bool(result or rhs)
        return result

    def _and_expr(self, values):
        result = self._unary(values)
        while self._peek() is not None and self._peek().upper() == 'AND':
            self._next()
            rhs = self._unary(values)
            result = bool(result and rhs)
        return result

    def _unary(self, values):
        tok = self._peek()
        if tok is not None and tok.upper() == 'NOT':
            self._next()
            return not self._unary(values)
        return self._primary(values)

    def _primary(self, values):
        tok = self._next()
        if tok is None:
            raise ValueError('逻辑表达式不完整')
        if tok == '(':
            result = self._or_expr(values)
            if self._next() != ')':
                raise ValueError('逻辑表达式括号不匹配')
            return result
        if tok not in values:
            raise ValueError(f'逻辑表达式中的因子 [{tok}] 未在配置中启用')
        return bool(values[tok])


# ======================== 多因子策略引擎 ========================

class EngineResult:
    """单只股票的多因子评估结果"""

    def __init__(self, code):
        self.code = code  # 股票代码
        self.passed = False  # 逻辑组合+评分门槛综合判定，是否入选
        self.score = 0.0  # 综合评分 0~1（命中因子权重和/总权重）
        self.matched_strategies = []  # 匹配的子策略列表（含NOT取反后的有效命中）
        self.detail = {}  # 各因子原始命中情况 {因子名: bool}
        self.logic_result = False  # 纯逻辑组合结果（未考虑评分门槛）

    def to_dict(self):
        return {
            'code': self.code,
            'passed': self.passed,
            'score': round(self.score, 4),
            'matched_strategies': '|'.join(self.matched_strategies),
            'logic_result': self.logic_result,
            'detail': self.detail,
        }

    def __repr__(self):
        return (f'<EngineResult {self.code} passed={self.passed} score={self.score:.2f} '
                f'matched={self.matched_strategies}>')


class MultiFactorEngine:
    """
    多因子组合选股引擎
    :param config: 多因子配置dict，默认读取 user_config.multi_factor，结构：
        {
            'factors': [{'name': '策略1', 'weight': 1.0, 'not': False, 'enabled': True}, ...],
            'logic': 'AND' | 'OR' | 表达式 "策略1 AND (策略2 OR NOT 卖策略)",
            'score_threshold': 0.0,  # 综合评分门槛(0~1)，0表示只看logic结果
        }
    """

    def __init__(self, config=None):
        if config is None:
            config = ucfg.multi_factor
        self.config = config
        # 只保留启用且已注册的因子
        self.factors = []
        for f in config.get('factors', []):
            if not f.get('enabled', True):
                continue
            if f['name'] not in STRATEGY_REGISTRY:
                print(f'[yellow]警告: 因子 [{f["name"]}] 未在策略注册表中注册，已跳过[/yellow]')
                continue
            self.factors.append(f)
        if not self.factors:
            raise ValueError('多因子配置中没有可用的因子，请检查 user_config.multi_factor')
        self.logic = str(config.get('logic', 'AND')).strip()
        self.score_threshold = float(config.get('score_threshold', 0.0))
        # logic 为简单关键字时转为表达式，统一走解析器。
        # not=True 的因子在自动生成的表达式中带 NOT 前缀，语义=“该因子不命中”
        if self.logic.upper() == 'AND':
            self._expression = ' AND '.join(
                ('NOT ' if f.get('not', False) else '') + f['name'] for f in self.factors)
        elif self.logic.upper() == 'OR':
            self._expression = ' OR '.join(
                ('NOT ' if f.get('not', False) else '') + f['name'] for f in self.factors)
        else:
            # 自定义表达式：表达式是逻辑组合的唯一取反入口（一律在因子原始命中值上求值），
            # 配置级 not 只影响评分口径与展示，不参与表达式求值，避免双重取反冲突
            self._expression = self.logic
            for f in self.factors:
                if f.get('not', False):
                    print(f"[yellow]提示: 因子 [{f['name']}] 配置级 not=True 仅用于评分口径，"
                          f"逻辑组合以表达式为准（如需取反请在表达式内写 NOT）[/yellow]")

    def factor_series(self, name, df, context):
        """计算指定因子的完整布尔序列（供监控/回测模块获取买卖信号序列使用）"""
        func, _ = STRATEGY_REGISTRY[name]
        series = func(df, context)
        return self._normalize(series, df.index)

    @staticmethod
    def _normalize(series, index):
        """统一因子输出为与df索引一致的布尔序列（fast模式返回单个bool时兜底处理）"""
        if isinstance(series, (bool, np.bool_)):
            result = pd.Series(False, index=index)
            if len(index) > 0:
                result.iloc[-1] = bool(series)
            return result
        if series is None or len(series) == 0:
            return pd.Series(False, index=index)
        series = series.reindex(index).fillna(False)
        return series.astype(bool)

    def evaluate_series(self, code, df, context=None, lookback=1):
        """
        对单只股票逐周期(bar)执行多因子组合评估，供监控模块对最近若干周期持续检测信号
        :param code: 股票代码
        :param df: 个股日线DF（时间为索引）
        :param context: 策略上下文，含HS300_信号/start_date/end_date等
        :param lookback: 只评估最后N个周期；None=评估全部周期
        :return: DataFrame，索引为周期日期，列含 passed/score/logic_result/matched_strategies/detail
        """
        if context is None:
            context = {}
        # 各因子的完整布尔序列
        factor_series = {}
        for f in self.factors:
            name = f['name']
            try:
                factor_series[name] = self.factor_series(name, df, context)
            except Exception as e:
                print(f'[yellow]{code} 因子[{name}]计算异常: {e}[/yellow]')
                factor_series[name] = pd.Series(False, index=df.index)
        weight_map = {f['name']: float(f.get('weight', 1.0)) for f in self.factors}
        negate_map = {f['name']: bool(f.get('not', False)) for f in self.factors}
        weight_sum = sum(weight_map.values())
        parser = _LogicParser(self._expression)  # 解析器复用，parse内部会重置pos
        bars = df.index if lookback is None else df.index[-lookback:]

        rows = []
        for bar in bars:
            raw = {name: bool(series.get(bar, False)) for name, series in factor_series.items()}
            # 逻辑组合：表达式在因子原始命中值上求值，NOT只能在表达式内书写
            logic_result = parser.parse(raw)
            # 综合评分：配置级not取反后的有效命中权重和 / 总权重
            score_sum = 0.0
            matched = []
            for name, hit in raw.items():
                effective = (not hit) if negate_map[name] else hit
                if effective:
                    score_sum += weight_map[name]
                    matched.append(('NOT ' if negate_map[name] else '') + name)
            score = (score_sum / weight_sum) if weight_sum > 0 else 0.0
            rows.append({'date': bar,
                         'passed': logic_result and score >= self.score_threshold,  # 逻辑通过且评分过门槛
                         'score': score,
                         'logic_result': logic_result,
                         'matched_strategies': '|'.join(matched),
                         'detail': raw})
        df_result = pd.DataFrame(rows)
        if len(df_result) == 0:
            return pd.DataFrame(columns=['date', 'passed', 'score', 'logic_result',
                                         'matched_strategies', 'detail'])
        return df_result.set_index('date', drop=False)

    def evaluate_stock(self, code, df, context=None):
        """
        对单只股票执行多因子组合评估（评估最后一个周期）
        :param code: 股票代码
        :param df: 个股日线DF（时间为索引）
        :param context: 策略上下文，含HS300_信号/start_date/end_date等
        :return: EngineResult 含匹配的子策略列表和综合评分
        """
        df_result = self.evaluate_series(code, df, context, lookback=1)
        result = EngineResult(code)
        if len(df_result) == 0:
            return result
        last = df_result.iloc[-1]
        result.passed = bool(last['passed'])
        result.score = float(last['score'])
        result.logic_result = bool(last['logic_result'])
        result.matched_strategies = last['matched_strategies'].split('|') if last['matched_strategies'] else []
        result.detail = dict(last['detail'])
        return result

    def evaluate_stocklist(self, stocklist, df_loader, context=None):
        """
        批量评估股票列表（单进程便捷接口，多进程场景请在子进程内构造引擎调用evaluate_stock）
        :param stocklist: 股票代码列表
        :param df_loader: 回调函数，入参code，返回该股日线DF
        :return: list[EngineResult]
        """
        results = []
        for code in stocklist:
            df = df_loader(code)
            if df is None or len(df) == 0:
                continue
            results.append(self.evaluate_stock(code, df, context))
        return results


if __name__ == '__main__':
    # 引擎自检测试：打印注册策略与当前多因子配置
    print('已注册策略:')
    for name, (_, desc) in STRATEGY_REGISTRY.items():
        print(f'  {name}: {desc}')
    engine = MultiFactorEngine()
    print(f'当前配置 因子: {[f["name"] for f in engine.factors]}')
    print(f'逻辑表达式: {engine._expression} 评分门槛: {engine.score_threshold}')
