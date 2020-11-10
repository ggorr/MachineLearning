import numpy as np


def getField(cls, value):
	for key in cls.__dict__.keys():
		if cls.__dict__[key] == value:
			return key


def getFloatStr(value: float) -> str:
	return np.format_float_scientific(value) if value < 1e-4 else np.format_float_positional(value)


def getClass(name: str):
	parts = name.split('.')
	c = __import__(".".join(parts[:-1]))
	for pkg in parts[1:]:
		c = getattr(c, pkg)
	return c
