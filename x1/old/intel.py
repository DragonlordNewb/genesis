from typing import Union, Iterable, Callable
from math import sqrt

Scalar = Union[int, float]

class VectorArithmeticError(ValueError):
	pass

badLength = VectorArithmeticError("Vectors must have same length to combine.")

class Vector:
	def __init__(self, *items: set[Scalar]):
		self.items = list(items)

	def __repr__(self) -> str:
		return "<" + ",".join([str(item) for item in self]) + ">"

	def __len__(self) -> int:
		return len(self.items)

	def __iter__(self) -> Iterable:
		return iter(self.items)

	def __add__(self, other: object) -> object:
		if len(self) != len(other):
			raise badLength
		return Vector(*[a + b for a, b in zip(self, other)])

	def __iadd__(self, other: object) -> None:
		if len(self) != len(other):
			raise badLength
		for index in range(len(self)):
			self.items[index] += other.items[index]

	def __sub__(self, other: object) -> object:
		if len(self) != len(other):
			raise badLength
		return Vector(*[a - b for a, b in zip(self, other)])

	def __isub__(self, other: object) -> None:
		if len(self) != len(other):
			raise badLength
		for index in range(len(self)):
			self.items[index] -= other.items[index]

	def __mul__(self, other: object) -> Scalar: 
		# dot product
		return sum([a * b for a, b in zip(self, other)])

	def __abs__(self) -> Scalar:
		return sqrt(sum([x ** 2 for x in self]))

	@classmethod
	def zeroes(cls, count: int) -> object:
		return cls(*[0 for _ in range(count)])

Number = Union[Scalar, Vector]

def dissolve(l: list) -> list:
	output = []
	for item in l:
		if type(item) == list:
			return 

def combinations(length: int, *data: set[any]) -> list[any]:
	data = list(data)
	if length == 1:
		return data
	return dissolve([[]])

class Action:
	def __init__(self, deltaFunction: Union[Callable, None]=None) -> None:
		if deltaFunction:
			self.delta = deltaFunction
		elif not hasattr(self, "delta"):
			raise SyntaxError("Must either provide a deltaFunction callable at initialization or subclass one onto the Action class.")

		self.sentiment = None

	def __rshift__(self, adaptor: None) -> Number:
		# application function
		#   Action >> Adapter
		oldEmotion = adaptor.computeEmotion()
		adaptor.state += self.delta(adaptor.state)
		newEmotion = adaptor.computeEmotion()
		improvement = newEmotion - oldEmotion

		if self.sentiment != None:
			self.sentiment += improvement
		else:
			self.sentiment = improvement

		return improvement

class ActionCombination:
	def __init__(self, *actions: set[Action]) -> None:
		self.actions = actions

	def __rshift__(self, adaptor: None) -> Number:
		return sum()

class ScalarCrossbarAdaptor:
	def __init__(self, ):
		pass