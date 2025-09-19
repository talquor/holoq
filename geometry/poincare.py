import numpy as np


EPS = 1e-9


# Poincaré ball with curvature c>0 (negative curvature -c)
class PoincareBall:
    def __init__(self, dim: int, c: float = 1.0):
        self.dim = dim
        self.c = float(c)


    def mobius_add(self, x, y):
        c = self.c
        x2 = np.sum(x * x, axis=-1, keepdims=True)
        y2 = np.sum(y * y, axis=-1, keepdims=True)
        xy = np.sum(x * y, axis=-1, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return num / np.maximum(den, EPS)


    def lambda_x(self, x):
        c = self.c
        x2 = np.sum(x * x, axis=-1, keepdims=True)
        return 2.0 / np.maximum(1.0 - c * x2, EPS)


    def distance(self, x, y):
        c = self.c
        mobius_minus = self.mobius_add(-x, y)
        norm = np.linalg.norm(mobius_minus, axis=-1)
        arg = 1 + 2 * c * (norm**2) / np.maximum((1 - c * np.sum(x * x, axis=-1)) * (1 - c * np.sum(y * y, axis=-1)), EPS)
        return np.arccosh(np.clip(arg, 1 + 1e-7, None)) / np.sqrt(c)


    def exp_map0(self, v):
        c = self.c
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        coef = np.tanh(np.sqrt(c) * v_norm) / (np.maximum(np.sqrt(c) * v_norm, EPS))
        return coef * v


    def log_map0(self, x):
        c = self.c
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
        coef = (1.0/np.sqrt(c)) * np.arctanh(np.sqrt(c) * x_norm) / np.maximum(x_norm, EPS)
        return coef * x