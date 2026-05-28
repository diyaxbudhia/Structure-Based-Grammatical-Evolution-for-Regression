import copy
from typing import List, Optional

import numpy as np

EPS = 1e-8

FUNCTION_SET = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "pdiv": 2,
    "sin": 1,
    "cos": 1,
    "log": 1,
    "exp": 1,
}


class Node:
    def __init__(self, kind: str, value=None, children: Optional[List["Node"]] = None):
        self.kind = kind
        self.value = value
        self.children = children or []

    def clone(self) -> "Node":
        return copy.deepcopy(self)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "var":
            return x[:, self.value]
        if self.kind == "const":
            return np.full(x.shape[0], self.value, dtype=float)

        values = [c.evaluate(x) for c in self.children]
        op = self.value

        if op == "add":
            out = values[0] + values[1]
        elif op == "sub":
            out = values[0] - values[1]
        elif op == "mul":
            out = values[0] * values[1]
        elif op == "pdiv":
            den = np.where(np.abs(values[1]) < EPS, EPS, values[1])
            out = values[0] / den
        elif op == "sin":
            out = np.sin(values[0])
        elif op == "cos":
            out = np.cos(values[0])
        elif op == "log":
            out = np.log(np.abs(values[0]) + EPS)
        elif op == "exp":
            out = np.exp(np.clip(values[0], -20.0, 20.0))
        else:
            raise ValueError(f"Unsupported op: {op}")

        return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def to_string(self) -> str:
        if self.kind == "var":
            return f"x{self.value}"
        if self.kind == "const":
            return f"{self.value:.4f}"

        op = self.value
        if op in {"add", "sub", "mul", "pdiv"}:
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            symbols = {"add": "+", "sub": "-", "mul": "*", "pdiv": "/"}
            return f"({left} {symbols[op]} {right})"
        return f"{op}({self.children[0].to_string()})"

    def structure_signature(self) -> str:
        if self.kind in {"var", "const"}:
            return "T"
        child_signatures = ",".join(c.structure_signature() for c in self.children)
        return f"{self.value}({child_signatures})"
