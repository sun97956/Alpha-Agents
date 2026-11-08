"""
因子表达式：校验与计算（PRD 第 7 节；接口见 contracts.py 第 4 部分 线 B）。

表达式是一行 Python 风格的字符串，只能由以下成分组成：
  - 字段：contracts.FIELDS
  - 时序算子：contracts.TS_OPS（最后一个参数是窗口，必须是整数字面量）
  - 截面算子：contracts.CS_OPS
  - 分组算子：group_neutralize(x, sector)，第二个参数固定为标识符 sector
  - 算术函数：log / abs / sign
  - 运算符：+ - * /、取负（-x）；括号
  - 数字常量（整数或小数）

不合规的表达式抛 ExprError，错误信息为中文，会直接返回给因子 Agent。
"""
from __future__ import annotations

import ast
from typing import Union

import numpy as np
import pandas as pd

from src import operators as ops
from src.contracts import (
    ARITH_FUNCS, COMPARE_OPS, COND_OPS, CS_OPS, DELAY_WINDOWS, FIELDS, GROUP_KEYS, GROUP_OPS,
    POWER_OPS, TS_OPS, WINDOWS,
    Market,
)

__all__ = ["ExprError", "validate", "compute"]

_DELAY_OPS = ("ts_delay", "ts_delta")
_ALL_FUNCS = {**TS_OPS, **CS_OPS, **GROUP_OPS, **ARITH_FUNCS, **POWER_OPS, **COND_OPS}
_CMP_SYMBOL = {ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">="}
_MAX_LEN = 2000
_MAX_DEPTH = 60

_BINOP_SYMBOL = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_BAD_BINOP_HINT = {
    ast.Pow: "**（乘方）", ast.Mod: "%（取余）", ast.FloorDiv: "//（整除）",
    ast.MatMult: "@", ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.LShift: "<<", ast.RShift: ">>",
}
_UNICODE_HINT = {
    "−": "-", "–": "-", "—": "-", "×": "*", "÷": "/", "·": "*",
    "（": "(", "）": ")", "，": ",",
}


class ExprError(ValueError):
    """表达式不合规。错误信息用中文写清楚原因。"""


def _fmt_set(xs) -> str:
    return "{" + ", ".join(str(x) for x in xs) + "}"


# ---------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------

def _parse(expr: str) -> ast.Expression:
    if not isinstance(expr, str):
        raise ExprError(f"公式必须是字符串，收到的类型是 {type(expr).__name__}。")
    s = expr.strip()
    if not s:
        raise ExprError("公式为空。")
    if len(s) > _MAX_LEN:
        raise ExprError(f"公式过长（{len(s)} 个字符），上限 {_MAX_LEN}。")
    bad = sorted({ch for ch in s if ch in _UNICODE_HINT})
    if bad:
        hint = "，".join(f"“{c}”请改为“{_UNICODE_HINT[c]}”" for c in bad)
        raise ExprError(f"公式里有全角或特殊符号：{hint}。只能使用 ASCII 的 + - * / ( ) ,。")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"公式语法错误（第 {e.offset} 个字符附近）：请检查括号、逗号和运算符是否完整。") from None
    except (ValueError, RecursionError, MemoryError):
        raise ExprError("公式无法解析：可能含有非法字符或嵌套过深。") from None
    return tree


def _window_value(node: ast.AST, op: str) -> int:
    allowed = DELAY_WINDOWS if op in _DELAY_OPS else WINDOWS
    if not (isinstance(node, ast.Constant) and type(node.value) is int):
        got = ast.unparse(node) if hasattr(ast, "unparse") else "?"
        raise ExprError(
            f"{op} 的最后一个参数是窗口，必须直接写一个整数，只能取 {_fmt_set(allowed)}；收到 “{got}”。"
        )
    d = node.value
    if d not in allowed:
        if d == 1 and op not in _DELAY_OPS:
            raise ExprError(
                f"{op} 的窗口不能取 1：只有 ts_delay 和 ts_delta 可以用 1。"
                f"{op} 的窗口只能取 {_fmt_set(WINDOWS)}。"
            )
        raise ExprError(f"{op} 的窗口只能取 {_fmt_set(allowed)}，收到 {d}。")
    return d


def _check(node: ast.AST, depth: int = 0) -> str:
    """递归校验，返回该节点的类型："series"（宽表）或 "const"（数字常量）。"""
    if depth > _MAX_DEPTH:
        raise ExprError(f"公式嵌套过深（超过 {_MAX_DEPTH} 层），请简化。")

    if isinstance(node, ast.Expression):
        return _check(node.body, depth + 1)

    if isinstance(node, ast.Constant):
        v = node.value
        if type(v) in (int, float):
            if not np.isfinite(v):
                raise ExprError(f"数字常量 {v!r} 不是有限数。")
            return "const"
        raise ExprError(f"不允许的常量 {v!r}：只能使用数字常量。")

    if isinstance(node, ast.Name):
        name = node.id
        if name in FIELDS:
            return "series"
        if name in GROUP_KEYS:
            raise ExprError(f"“{name}” 只能作为 group_neutralize 的第二个参数，例如 group_neutralize(x, {name})。")
        if name in _ALL_FUNCS:
            raise ExprError(f"“{name}” 是算子，必须带括号和参数调用。")
        raise ExprError(f"不允许的字段或标识符 “{name}”。可用字段只有：{', '.join(FIELDS)}。")

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return _check(node.operand, depth + 1)
        sym = {ast.UAdd: "+（正号）", ast.Not: "not", ast.Invert: "~"}.get(type(node.op), type(node.op).__name__)
        raise ExprError(f"不允许的一元运算符 {sym}：只允许取负（-x）。")

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOP_SYMBOL:
            sym = _BAD_BINOP_HINT.get(type(node.op), type(node.op).__name__)
            raise ExprError(f"不允许的运算符 {sym}：只能使用 + - * /。")
        lk = _check(node.left, depth + 1)
        rk = _check(node.right, depth + 1)
        return "series" if "series" in (lk, rk) else "const"

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _CMP_SYMBOL:
            raise ExprError("比较运算只能写一个 < > <= >=，例如 where(volume > ts_mean(volume, 20), x, y)；不支持 == != 和连写。")
        lk = _check(node.left, depth + 1)
        rk = _check(node.comparators[0], depth + 1)
        if "series" not in (lk, rk):
            raise ExprError("比较运算两边不能都是数字常量。")
        return "series"

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExprError("不允许的调用形式：只能直接调用白名单算子，例如 ts_mean(close, 20)，不能用属性或方法。")
        op = node.func.id
        if op not in _ALL_FUNCS:
            allowed = ", ".join(list(TS_OPS) + list(CS_OPS) + list(GROUP_OPS) + list(ARITH_FUNCS)
                                  + list(POWER_OPS) + list(COND_OPS))
            raise ExprError(f"不允许的算子 “{op}”。可用算子只有：{allowed}。")
        if node.keywords:
            raise ExprError(f"{op} 不支持关键字参数（如 d=20），请按位置传参。")
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise ExprError(f"{op} 不支持 * 展开参数。")
        args = node.args

        if op in TS_OPS:
            n_data = TS_OPS[op]
            if len(args) != n_data + 1:
                sig = ", ".join(["x", "y"][:n_data] + ["d"])
                raise ExprError(f"{op} 需要 {n_data + 1} 个参数：{op}({sig})，收到 {len(args)} 个。")
            for a in args[:n_data]:
                _need_series(_check(a, depth + 1), op, a)
            _window_value(args[-1], op)
            return "series"

        if op in CS_OPS or op in ARITH_FUNCS:
            n = CS_OPS.get(op, ARITH_FUNCS.get(op))
            if len(args) != n:
                raise ExprError(f"{op} 需要 {n} 个参数：{op}(x)，收到 {len(args)} 个。")
            _need_series(_check(args[0], depth + 1), op, args[0])
            return "series"

        if op in POWER_OPS:
            if len(args) != 2:
                raise ExprError(f"power 需要 2 个参数：power(x, a)，a 为数字常量，收到 {len(args)} 个。")
            _need_series(_check(args[0], depth + 1), op, args[0])
            if _check(args[1], depth + 1) != "const":
                raise ExprError("power 的第二个参数（指数）必须是数字常量。")
            return "series"

        if op in COND_OPS:
            if len(args) != 3:
                raise ExprError(f"where 需要 3 个参数：where(条件, 条件成立时的值, 否则的值)，收到 {len(args)} 个。")
            kinds = [_check(a, depth + 1) for a in args]
            _need_series(kinds[0], op, args[0])
            return "series"

        if op in GROUP_OPS:
            if len(args) != GROUP_OPS[op] + 1:
                raise ExprError(f"{op} 需要 2 个参数：{op}(x, sector)，收到 {len(args)} 个。")
            _need_series(_check(args[0], depth + 1), op, args[0])
            g = args[1]
            if not (isinstance(g, ast.Name) and g.id in GROUP_KEYS):
                got = ast.unparse(g)
                raise ExprError(f"{op} 的第二个参数只能是 {' / '.join(GROUP_KEYS)}，收到 “{got}”。")
            return "series"

    raise ExprError(
        f"不允许的语法：{_describe(node)}。只能使用字段、白名单算子、数字常量、+ - * /、取负和括号。"
    )


def _need_series(kind: str, op: str, node: ast.AST) -> None:
    if kind != "series":
        raise ExprError(f"{op} 的数据参数必须包含字段，不能只是数字常量（收到 “{ast.unparse(node)}”）。")


def _describe(node: ast.AST) -> str:
    names = {
        ast.Compare: "比较运算（只支持单个 < > <= >=）", ast.BoolOp: "逻辑运算（and / or，请改用嵌套 where）",
        ast.IfExp: "条件表达式（if else）", ast.Attribute: "属性访问（.xxx）",
        ast.Subscript: "下标（[...]）", ast.Lambda: "lambda", ast.List: "列表",
        ast.Tuple: "元组", ast.Dict: "字典", ast.Set: "集合",
        ast.JoinedStr: "f-string", ast.NamedExpr: "海象运算符（:=）",
    }
    return names.get(type(node), type(node).__name__)


def validate(expr: str) -> None:
    """不合规则抛 ExprError（中文原因）；合规则返回 None。"""
    tree = _parse(expr)
    try:
        kind = _check(tree)
    except RecursionError:
        raise ExprError("公式嵌套过深，请简化。") from None
    if kind != "series":
        raise ExprError("公式里没有任何字段，结果是常数，不能作为因子。")


# ---------------------------------------------------------------------
# 计算
# ---------------------------------------------------------------------

Value = Union[pd.DataFrame, float]


def _eval(node: ast.AST, market: Market) -> Value:
    if isinstance(node, ast.Expression):
        return _eval(node.body, market)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        # 时序计算在完整历史上进行，这里不按股票池截断
        return market.fields[node.id].astype("float64")
    if isinstance(node, ast.UnaryOp):          # 只有 USub
        v = _eval(node.operand, market)
        return -v
    if isinstance(node, ast.BinOp):
        a = _eval(node.left, market)
        b = _eval(node.right, market)
        if isinstance(node.op, ast.Add):
            r = a + b
        elif isinstance(node.op, ast.Sub):
            r = a - b
        elif isinstance(node.op, ast.Mult):
            r = a * b
        else:
            return ops.safe_div(a, b)
        return ops.clean(r) if isinstance(r, pd.DataFrame) else ops._clean_scalar(r)
    if isinstance(node, ast.Compare):
        a = _eval(node.left, market)
        b = _eval(node.comparators[0], market)
        return ops.compare(a, b, _CMP_SYMBOL[type(node.ops[0])])
    if isinstance(node, ast.Call):
        op = node.func.id
        args = node.args
        if op == "power":
            return ops.power(_eval(args[0], market), float(_eval(args[1], market)))
        if op == "where":
            return ops.where(*[_eval(a, market) for a in args])
        if op in TS_OPS:
            n = TS_OPS[op]
            data = [_eval(a, market) for a in args[:n]]
            d = int(args[-1].value)
            return getattr(ops, op)(*data, d)
        if op == "cs_rank":
            return ops.cs_rank(_eval(args[0], market), market.universe)
        if op == "cs_zscore":
            return ops.cs_zscore(_eval(args[0], market), market.universe)
        if op == "group_neutralize":
            return ops.group_neutralize(_eval(args[0], market), market.group(args[1].id), market.universe)
        if op == "log":
            return ops.log(_eval(args[0], market))
        if op == "abs":
            return ops.abs_(_eval(args[0], market))
        if op == "sign":
            return ops.sign(_eval(args[0], market))
    raise ExprError(f"内部错误：无法计算节点 {_describe(node)}。")  # validate 之后不应到达


def compute(expr: str, market: Market) -> pd.DataFrame:
    """
    校验后计算因子值。返回与 market 同形状的 float64 宽表；
    只保留 universe 为 True 的位置，其余为 NaN；inf -> NaN。
    """
    validate(expr)
    tree = _parse(expr)
    v = _eval(tree, market)
    if not isinstance(v, pd.DataFrame):          # validate 已保证不会是常数
        raise ExprError("公式结果是常数，不能作为因子。")
    close = market.fields["close"]
    v = v.reindex(index=close.index, columns=close.columns)
    u = market.universe.reindex(index=close.index, columns=close.columns, fill_value=False).astype(bool)
    return ops.clean(v).where(u)
